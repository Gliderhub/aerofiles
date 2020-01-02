import math
import datetime as dt
import time
import re
import numpy as np

from aerofiles.util.geo import METER_PER_FEET
from .config import FlightParsingConfig

try:
    from django.core.exceptions import ValidationError
except ImportError:
    class ValidationError(Exception):
        pass


class Reader:
    def __init__(self, file, config_class=FlightParsingConfig):
        self.file = file
        self.config = config_class
        self.igc_file = {}
        self.records = {
            'A': [],
            'B': [],
            'C': [],
            'I': [],
            'H': []
        }
        self.notes = []

    def read(self):
        """
        Creates a dictionary with different records
        """
        # iterate the lines and append to the corresponding record
        # for line in self.file.decode("ISO-8859-1") :
        for line in self.file:
            line = line.replace('\n', '').replace('\r', '')
            if not line:
                continue
            if line[0] == 'A':
                self.records['A'].append(line)
            elif line[0] == 'B':
                self.records['B'].append(line)
            elif line[0] == 'C':
                self.records['C'].append(line)
            elif line[0] == 'I':
                self.records['I'].append(line)
            elif line[0] == 'H':
                self.records['H'].append(line)
            else:
                # Do not parse any other types of IGC records
                pass
        return self.records

    def validate(self):
        self.validate_a_record()
        self.validate_h_record()
        self.validate_i_record()
        self.validate_b_record()

        return self.igc_file, self.fixes, self.notes

    def validate_a_record(self):
        if len(self.records['A']) > 1:
            self.notes.append('Multiple A Records provided')
        if not self.records['A']:
            self.notes.append('No A Record provided.')
        if len(self.records['A'][0]) >= 7:
            self.igc_file['fr_manuf_id'] = (
                LowLevelReader.strip_non_printable_chars(self.records['A'][0][1:4]))
            self.igc_file['fr_uniq_id'] = (
                LowLevelReader.strip_non_printable_chars(self.records['A'][0][4:7]))
        else:
            self.notes.append('A Record is too short ')

    def validate_h_record(self):
        """
        Check that HFDTE record exists and contains valid date
        """
        # at least the date must be provided
        if not self.records['H']:
            self.notes.append('No H Records provided')
        for line in self.records['H']:
            source = line[1]

            tlc = line[2:5] # three letter code
            if tlc == 'DTE':
                # TODO .strftime("%Y-%m-%d %H:%M:%S")
                self.igc_file['date_utc'] = LowLevelReader.decode_H_utc_date(line)
            elif tlc == 'FXA':
                self.igc_file['fix_accuracy'] = LowLevelReader.decode_H_fix_accuracy(line)
            elif tlc == 'PLT':
                self.igc_file['pilot_name'] = LowLevelReader.decode_H_pilot(line)
            elif tlc == 'CM2':
                self.igc_file['copilot_name'] = LowLevelReader.decode_H_copilot(line)
            elif tlc == 'GTY':
                self.igc_file['model_name'] = LowLevelReader.decode_H_glider_model(line)
            elif tlc == 'GID':
                self.igc_file['registration'] = LowLevelReader.decode_H_glider_registration(line)
            elif tlc == 'DTM':
                self.gps_date = LowLevelReader.decode_H_gps_datum(line)
            elif tlc == 'RFW':
                self.igc_file['fr_firmware_version'] = LowLevelReader.decode_H_firmware_revision(line)
            elif tlc == 'RHW':
                self.igc_file['fr_hardware_version'] = LowLevelReader.decode_H_hardware_revision(line)
            elif tlc == 'FTY':
                self.igc_file['fr_manuf_name'], self.igc_file['fr_name'] = LowLevelReader.decode_H_manufacturer_model(line)
            elif tlc == 'GPS':
                gps_info = LowLevelReader.decode_H_gps_receiver(line)
                self.igc_file['fr_gps_model'] = gps_info['gps_model']
                self.igc_file['fr_gps_channels'] = gps_info['gps_channels']
                self.igc_file['gps_max_alt'] = gps_info['gps_max_alt']
            elif tlc == 'PRS':
                pressure_info = LowLevelReader.decode_H_pressure_sensor(line)
                self.igc_file['fr_pressure_sensor'] = pressure_info['pressure_sensor_model']
                self.igc_file['pressure_max_alt'] = pressure_info['pressure_sensor_max_alt']
            elif tlc == 'CID':
                self.igc_file['competition_id'] = LowLevelReader.decode_H_competition_id(line)
            elif tlc == 'CCL':
                self.igc_file['competition_class'] = LowLevelReader.decode_H_competition_class(line)
            elif tlc == 'TZN':
                self.igc_file['time_zone_offset'] = LowLevelReader.decode_H_time_zone_offset(line)
            elif tlc == 'MOP':
                # TOTHINK: Do we need this record to choose sensor?
                self.mop_sensor = LowLevelReader.decode_H_mop_sensor(line)
                # self.igc_file['mop_sensor'] = LowLevelReader.decode_H_mop_sensor(line)
            elif tlc == 'SIT':
                self.site = LowLevelReader.decode_H_site(line)
            elif tlc == 'TZO':
                self.igc_file['time_zone_offset'] = LowLevelReader.decode_H_time_zone_offset(line)
            elif tlc == 'UNT':
                self.units = LowLevelReader.decode_H_units_of_measure(line)
            else:
                self.notes.append(f'Invalid H record code: {tlc}')
        # date must be provided
        if 'date_utc' not in self.igc_file:
            self.notes.append('No date in H Record provided')

    def validate_i_record(self):
        """
        Validates the IGC I Records.

        I records contain a description of extensions used in B records.
        Available sensors for detecting engine run are checked against the
        preference of parser.
        """
        if not self.records['I']:
            raise ValidationError('No I records provided')
        if len(self.records['I']) > 1:
            raise ValidationError('Multiple I records provided')

        self.extensions = LowLevelReader.decode_extension_record(self.records['I'][0])
        self.igc_file['extension_types'] = (
            [s['extension_type'] for s in self.extensions]
        )

    def validate_b_record(self):

        self.build_fixes()
        self.validate_altitudes()
        self.validate_fix_raw_time()

        if not(self.pressure_alt_valid or self.gnss_alt_valid):
            self.valid = False
            self.notes.append(
                'Error: Neither pressure nor gnss altitude is valid.'
            )
        if self.fixes[-1]['time'] > dt.datetime.utcnow():
            self.valid = False
            self.notes.append('Flight date is from future')

    def build_fixes(self):
        self.fixes = []

        if not self.records['B']:
            raise ValidationError('No B Records found')
        if len(self.records['B']) < self.config.min_fixes:
            raise ValidationError(
                f"Error: This file has {len(self.records['B'])} fixes, less "
                f'than the minimum {self.config.min_fixes}.'
            )
        self.date = self.igc_file.get('date_utc')
        if self.date is None:
            self.date = dt.datetime.utcnow().date()
        for line in self.records['B']:
            try:
                fix = self.build_fix(line)
            except (ValueError, TypeError):
                continue
            if fix is not None:
                if self.fixes and np.abs(fix['raw_time']-self.fixes[-1]['raw_time']) < 1e-5:
                    # The time did not change since the previous fix.
                    # Ignore this fix.
                    pass
                else:
                    self.fixes.append(fix)

    def build_fix(self, line):
        """Creates GNSSFix object from IGC B-record line.
        Args:
            B_record_line: a string, B record line from an IGC file
            index: the zero-based position of the fix in the parent IGC file
        Returns:
            The created GNSSFix object
        """
        match = re.match(
            '^B' + '(\d\d)(\d\d)(\d\d)'
            + '(\d\d)(\d\d)(\d\d\d)([NS])'
            + '(\d\d\d)(\d\d)(\d\d\d)([EW])'
            + '([AV])' + '([-\d]\d\d\d\d)' + '([-\d]\d\d\d\d)'
            + '([0-9a-zA-Z\-]*).*$', line)
        if match is None:
            return None
        (hours, minutes, seconds,
         lat_deg, lat_min, lat_min_dec, lat_sign,
         lon_deg, lon_min, lon_min_dec, lon_sign,
         validity, press_alt, gnss_alt,
         ext) = match.groups()

        time = dt.datetime(
            year=self.date.year,
            month=self.date.month,
            day=self.date.day,
            hour=int(hours),
            minute=int(minutes),
            second=int(seconds)
        )
        raw_time = ((time.hour*60)+time.minute)*60 + time.second

        lat = float(lat_deg)
        lat += float(lat_min) / 60.0
        lat += float(lat_min_dec) / 1000.0 / 60.0
        if lat_sign == 'S':
            lat = -lat

        lon = float(lon_deg)
        lon += float(lon_min) / 60.0
        lon += float(lon_min_dec) / 1000.0 / 60.0
        if lon_sign == 'W':
            lon = -lon

        press_alt = float(press_alt)
        gnss_alt = float(gnss_alt)

        fix = {
            'time': time,
            'raw_time': raw_time,
            'lon': lon,
            'lat': lat,
            'pressure_alt': press_alt,
            'gnss_alt': gnss_alt,
        }

        for extension in self.extensions:
            start_byte, end_byte = extension['bytes']
            start_byte = start_byte - 35 - 1
            end_byte = end_byte - 35 - 1

            if extension['extension_type'] == 'FXA':
                # Fix accuracy. When used in the B (fix) record, this is the EPE
                # (Estimated Position Error) figure in metres (MMM) for the
                # individual fix concerned, to a 2-Sigma (95.45%) probability
                 fix['fxa'] = int(ext[start_byte:end_byte + 1])

            if extension['extension_type'] == 'ENL':
                # Environmental Noise Level. The ENL system is inside the FR and is
                # intended to record when an engine is running in three numbers
                # between 000 and 999 in the fix records of the IGC file.
                fix['enl'] = int(ext[start_byte:end_byte + 1])

            if extension['extension_type'] == 'RPM':
                # Revolutions Per Minute (of engine)
                fix['rpm'] = int(ext[start_byte:end_byte + 1])

            if extension['extension_type'] == 'MOP':
                # Means of Propulsion. A signal from an engine-related function
                # from a sensor connected by cable to the FR and placed close to
                # the engine and/or propeller, giving three numbers between 000 and
                # 999 in the fix records of the IGC file.
                fix['mop'] = int(ext[start_byte:end_byte + 1])

            if extension['extension_type'] == 'TAS':
                fix['tas'] = float(ext[start_byte:end_byte + 1]) / 100

            if extension['extension_type'] == 'VAT':
                # Compensated variometer (total energy/NETTO) vertical speed in
                # metres per second and tenths of metres per second with
                # leading zero and no dot (".") separator between metres and
                # tenths. Valid characters 0-9 and negative sign "-". Negative
                # values to have negative sign instead of leading zero
                fix['vat'] = float(ext[start_byte:end_byte + 1]) / 100

            if extension['extension_type'] == 'WDI':
                # Wind Direction (the direction the wind is coming from).
                # Three numbers based on degrees clockwise from 000 for north
                fix['wdi'] = int(ext[start_byte:end_byte + 1])

            if extension['extension_type'] == 'WSP':
                # Wind speed, three numbers in kilometres per hour
                fix['wsp'] = int(ext[start_byte:end_byte + 1])

            if extension['extension_type'] == 'CUR':
                # Electrical current, Amperes
                fix['cur'] = int(ext[start_byte:end_byte + 1])

            if extension['extension_type'] == 'VOL':
                # Electrical Volts
                fix['cur'] = int(ext[start_byte:end_byte + 1])
        return fix

    def validate_altitudes(self):
        pressure_alt_violations_num = 0
        gnss_alt_violations_num = 0
        pressure_huge_changes_num = 0
        gnss_huge_changes_num = 0
        pressure_chgs_sum = 0.0
        gnss_chgs_sum = 0.0
        for i in range(len(self.fixes) - 1):
            pressure_alt_delta = math.fabs(
                self.fixes[i+1]['pressure_alt'] - self.fixes[i]['pressure_alt'])
            gnss_alt_delta = math.fabs(
                self.fixes[i+1]['gnss_alt'] - self.fixes[i]['gnss_alt'])
            raw_time_delta = math.fabs(
                self.fixes[i+1]['raw_time'] - self.fixes[i]['raw_time'])
            if raw_time_delta > 0.5:
                if (pressure_alt_delta / raw_time_delta >
                        self.config.max_alt_change_rate):
                    pressure_huge_changes_num += 1
                else:
                    pressure_chgs_sum += pressure_alt_delta
                if (gnss_alt_delta / raw_time_delta >
                        self.config.max_alt_change_rate):
                    gnss_huge_changes_num += 1
                else:
                    gnss_chgs_sum += gnss_alt_delta
            if (self.fixes[i]['pressure_alt'] > self.config.max_alt
                    or self.fixes[i]['pressure_alt'] < self.config.min_alt):
                pressure_alt_violations_num += 1
            if (self.fixes[i]['gnss_alt'] > self.config.max_alt or
                    self.fixes[i]['gnss_alt'] < self.config.min_alt):
                gnss_alt_violations_num += 1
        pressure_chgs_avg = pressure_chgs_sum / float(len(self.fixes) - 1)
        gnss_chgs_avg = gnss_chgs_sum / float(len(self.fixes) - 1)

        pressure_alt_ok = True
        if pressure_chgs_avg < self.config.min_avg_abs_alt_change:
            self.notes.append(
                f'Warning: average pressure altitude change between fixes '
                f'is: {pressure_chgs_avg}. It is lower than the minimum: '
                f'{self.config.min_avg_abs_alt_change}.'
            )
            pressure_alt_ok = False

        if pressure_huge_changes_num > self.config.max_alt_change_violations:
            self.notes.append(
                f'Warning: Too many high changes in pressure altitude: '
                f'{pressure_huge_changes_num}. '
                f'Maximum allowed: {self.config.max_alt_change_violations}.'
            )
            pressure_alt_ok = False

        if pressure_alt_violations_num > 0:
            self.notes.append(
                f'Warning: Pressure altitude limits exceeded in '
                f'{pressure_alt_violations_num} fixes.'
            )
            pressure_alt_ok = False

        gnss_alt_ok = True
        if gnss_chgs_avg < self.config.min_avg_abs_alt_change:
            self.notes.append(
                f'Warning: average gnss altitude change between fixes is: '
                f'{gnss_chgs_avg}. It is lower than the minimum: '
                f'{self.config.min_avg_abs_alt_change}.'
            )
            gnss_alt_ok = False

        if gnss_huge_changes_num > self.config.max_alt_change_violations:
            self.notes.append(
                f'Warning: Too many high changes in gnss altitude: {gnss_huge_changes_num}. '
                f'Maximum allowed: {self.config.max_alt_change_violations}.'
            )
            gnss_alt_ok = False

        if gnss_alt_violations_num > 0:
            self.notes.append(
                f'Warning: GNSS altitude limits exceeded in '
                f'{gnss_alt_violations_num} fixes.'
                )
            gnss_alt_ok = False
        self.pressure_alt_valid = pressure_alt_ok
        self.gnss_alt_valid = gnss_alt_ok

    def validate_fix_raw_time(self):
        """
        Checks for raw_time anomalies, fixes 0:00 UTC crossing.
        The B records do not have fully qualified time_stamps (just the current
        time in UTC), therefore flights that cross 0:00 UTC need special
        handling.
        """
        DAY = 24*60*60
        days_added = 0
        raw_time_between_fix_exceeded = 0
        for i in range(1, len(self.fixes)):
            f0 = self.fixes[i-1]
            f1 = self.fixes[i]
            f1['time'] += dt.timedelta(days=days_added)
            f1['raw_time'] += days_added * DAY

            if (f0['raw_time'] > f1['raw_time'] and
                    f1['raw_time'] + DAY < f0['raw_time'] + 200.0):
                # Day switch
                days_added += 1
                f1['time'] += dt.timedelta(days=days_added)
                f1['raw_time'] += days_added * DAY

            time_change = f1['raw_time'] - f0['raw_time']
            if time_change < self.config.min_seconds_between_fixes - 1e-5:
                raw_time_between_fix_exceeded += 1
            if time_change > self.config.max_seconds_between_fixes + 1e-5:
                raw_time_between_fix_exceeded += 1

        if raw_time_between_fix_exceeded > self.config.max_time_violations:
            self.valid = False
            self.notes.append(
                f'Error: Too many fixes intervals exceed time between fixes '
                f'constraints. Allowed {self.config.max_time_violations} '
                f'fixes, found {raw_time_between_fix_exceeded} fixes.'
            )
        if days_added > self.config.max_new_days_in_flight:
            self.valid = False
            self.notes.append(
                f'Error: Too many times did the flight cross the UTC 0:00 '
                f'barrier. Allowed {self.config.max_new_days_in_flight} '
                f'times, found {days_added} times.'
            )


class LowLevelReader:
    """
    A low level reader for the IGC flight log file format.
    see http://carrier.csi.cam.ac.uk/forsterlewis/soaring/igc_file_format/igc_format_2008.html
    """

    @staticmethod
    def strip_non_printable_chars(string):
        """Filters a string removing non-printable characters.
        Args:
            string: A string to be filtered.
        Returns:
            A string, where non-printable characters are removed.
        """
        printable = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKL"
                        "MNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ ")

        printable_string = [x for x in string if x in printable]
        return ''.join(printable_string)

    @staticmethod
    def rawtime_float_to_hms(timef):
        """Converts time from floating point seconds to hours/minutes/seconds.
        Args:
            timef: A floating point time in seconds to be converted
        Returns:
            A namedtuple with hours, minutes and seconds elements
        """
        time = int(round(timef))
        hms = collections.namedtuple('hms', ['hours', 'minutes', 'seconds'])

        return hms((time/3600), (time % 3600)/60, time % 60)


    @staticmethod
    def decode_H_utc_date(line):
        match = re.search(r'\d{6}', line)
        if match is None:
            return None
        return LowLevelReader.decode_date(match.group())

    @staticmethod
    def decode_H_fix_accuracy(line):
        fix_accuracy = line[5:].strip()
        return None if fix_accuracy == '' else int(fix_accuracy)

    @staticmethod
    def decode_H_pilot(line):
        pilot = line[line.find(':') + 1:].strip()
        return None if pilot == '' else pilot

    @staticmethod
    def decode_H_copilot(line):
        second_pilot = line[11:].strip()
        return None if second_pilot == '' else second_pilot

    @staticmethod
    def decode_H_glider_model(line):
        glider_model = line[16:].strip()
        return None if glider_model == '' else glider_model

    @staticmethod
    def decode_H_glider_registration(line):
        glider_registration = line[14:].strip()
        if glider_registration == '':
            return None
        else:
            return glider_registration

    @staticmethod
    def decode_H_gps_datum(line):
        gps_datum = line[17:].strip()
        return None if gps_datum == '' else gps_datum

    @staticmethod
    def decode_H_firmware_revision(line):
        firmware_revision = line[21:].strip()
        return None if firmware_revision == '' else firmware_revision

    @staticmethod
    def decode_H_hardware_revision(line):
        hardware_revision = line[21:].strip()
        return None if hardware_revision == '' else hardware_revision

    @staticmethod
    def decode_H_manufacturer_model(line):
        manufacturer = None
        model = None
        manufacturer_model = line[12:].strip().split(',')
        if manufacturer_model[0] != '':
            manufacturer = manufacturer_model[0].strip()
        if len(manufacturer_model) == 2 and manufacturer_model[1].lstrip() != '':
            model = manufacturer_model[1].strip()
        return manufacturer, model

    @staticmethod
    def decode_H_gps_receiver(line):

        # some IGC files use colon, others don't
        if line[5] == ':':
            gps_sensor = line[6:].lstrip().split(',')
        else:
            gps_sensor = line[5:].split(',')

        manufacturer = None
        model = None
        channels = None
        max_alt = None
        for detail_index, detail in enumerate(reversed(gps_sensor)):
            if len(gps_sensor) == 1 or (len(gps_sensor) == 2 and gps_sensor[1].strip() == ''):
                manufacturer = detail.strip()
            elif len(gps_sensor) == 3:
                if detail_index == 0:
                    max_alt = detail.strip()
                elif detail_index == 1:
                    channels = detail.strip()
                else:  # detail_index == 2
                    manufacturer = detail.strip()
            elif len(gps_sensor) == 4:
                if detail_index == 0:
                    max_alt = detail.strip()
                elif detail_index == 1:
                    channels = detail.strip()
                elif detail_index == 2:
                    model = detail.strip()
                else:  # detail_index == 3
                    manufacturer = detail.strip()
            else:
                raise ValidationError

        # stripping of ch from '12ch'
        if channels is not None:
            if channels.endswith('ch') or channels.endswith('Ch') or channels.endswith('CH'):
                channels = int(channels[:-2])
            else:
                channels = int(channels)

        # stripping of max from 'max10000m'
        if max_alt is not None and max_alt.startswith('max'):
            max_alt = max_alt[3::]

        # separate unit from value
        if max_alt is not None and max_alt.endswith('m'):
            max_alt_value = int(max_alt[:-1])
        elif max_alt is not None and max_alt.endswith('ft'):
            max_alt_value = int(max_alt[:-2]) * METER_PER_FEET
        elif max_alt is not None:
            max_alt_value = int(max_alt)
        else:
            max_alt_value = None

        return {
            'gps_manufacturer': manufacturer,
            'gps_model': model,
            'gps_channels': channels,
            'gps_max_alt': max_alt_value
        }

    @staticmethod
    def decode_H_pressure_sensor(line):

        manufacturer = None
        model = None
        max_alt = None

        # some IGC files use colon, others don't
        if line[19] == ':':
            pressure_sensor = line[20:].strip().split(',')
        else:
            pressure_sensor = line[19:].split(',')

        if len(pressure_sensor) == 1:
            manufacturer = pressure_sensor[0].strip() if pressure_sensor[0] != '' else None
        elif len(pressure_sensor) == 2:
            manufacturer_model = pressure_sensor[0].strip().split(' ') if pressure_sensor[0] != '' else None

            if len(manufacturer_model) == 2:
                manufacturer = manufacturer_model[0]
                model = manufacturer_model[1]
            else:
                manufacturer = manufacturer_model[0]

            max_alt = pressure_sensor[1].strip() if pressure_sensor[1] != '' else None
        elif len(pressure_sensor) == 3:
            manufacturer = pressure_sensor[0].strip() if pressure_sensor[0] != '' else None
            model = pressure_sensor[1].strip() if pressure_sensor[1] != '' else None
            max_alt = pressure_sensor[2].strip() if pressure_sensor[2] != '' else None

        # stripping of max from 'max10000m'
        if max_alt is not None and max_alt.startswith('max'):
            max_alt = max_alt[3::]

        # separate unit from value
        if max_alt is not None and max_alt.endswith('m'):
            max_alt_value = int(max_alt[:-1])
        elif max_alt is not None and max_alt.endswith('ft'):
            max_alt_value = int(max_alt[:-2]) * METER_PER_FEET
        elif max_alt is not None:
            max_alt_value = int(max_alt)
        else:
            max_alt_value = None

        return {
            'pressure_sensor_manufacturer': manufacturer,
            'pressure_sensor_model': model,
            'pressure_sensor_max_alt': max_alt_value
        }

    @staticmethod
    def decode_H_competition_id(line):
        competition_id = line[19:].strip()
        return None if competition_id == '' else competition_id

    @staticmethod
    def decode_H_competition_class(line):
        competition_class = line[22:].strip()
        return None if competition_class == '' else competition_class

    @staticmethod
    def decode_H_time_zone_offset(line):
        return int(float(line[14::].strip()))

    @staticmethod
    def decode_H_mop_sensor(line):
        return line[12::].strip()

    @staticmethod
    def decode_H_site(line):
        return line[10::].strip()

    @staticmethod
    def decode_H_units_of_measure(line):
        return line[11::].strip().split(',')

    @staticmethod
    def decode_date(date_str):
        if not date_str.isdigit() or len(date_str) != 6 or date_str == '000000':
            return None

        dd = int(date_str[0:2])
        mm = int(date_str[2:4])
        yy = int(date_str[4:6])

        current_year_yyyy = dt.date.today().year
        current_year_yy = current_year_yyyy % 100
        current_century = current_year_yyyy - current_year_yy
        yyyy = current_century + yy if yy <= current_year_yy else current_century - 100 + yy

        return dt.date(yyyy, mm, dd)

    @staticmethod
    def decode_time(time_str):
        if not date_str.isdigit() or len(date_str) != 6 or date_str == '000000':
            return None

        hh = int(time_str[0:2])
        mm = int(time_str[2:4])
        ss = int(time_str[4:6])

        return dt.time(hh, mm, ss)

    @staticmethod
    def decode_datetime(dt_str):
        if len(dt_str) != 12 or not dt_str[0:6].isdigit() or not dt_str[6:12].isdigit():
            return None

        dd = int(dt_str[0:2])
        mm = int(dt_str[2:4])
        yy = int(dt_str[4:6])

        hh = int(dt_str[6:8])
        MM = int(dt_str[8:10])
        ss = int(dt_str[10:12])

        current_year_yyyy = dt.date.today().year
        current_year_yy = current_year_yyyy % 100
        current_century = current_year_yyyy - current_year_yy
        yyyy = current_century + yy if yy <= current_year_yy else current_century - 100 + yy
        try:
            return dt.datetime(yyyy, mm, dd, hh, MM, ss)
        except:
            return None

    @staticmethod
    def decode_extension_record(line):

        no_extensions = int(line[1:3])

        if no_extensions * 7 + 3 != len(line.strip()):
            raise ValueError('I record contains incorrect number of digits')

        extensions = []
        for extension_index in range(no_extensions):
            extension_str = line[extension_index * 7 + 3:(extension_index + 1) * 7 + 3]
            start_byte = int(extension_str[0:2])
            end_byte = int(extension_str[2:4])
            tlc = extension_str[4:7]

            extensions.append({'bytes': (start_byte, end_byte), 'extension_type': tlc})

        return extensions

    @staticmethod
    def decode_latitude(lat_string):

        d = int(lat_string[0:2])
        m = float(lat_string[2:7]) / 1000
        ordinal = lat_string[7]

        latitude = d + m / 60.

        if not (0. <= latitude <= 90):
            raise ValueError('Latitude format is invalid')

        if ordinal == 'S':
            latitude = -latitude

        return latitude

    @staticmethod
    def decode_longitude(lon_string):

        d = float(lon_string[0:3])
        m = float(lon_string[3:8]) / 1000
        ordinal = lon_string[8]

        longitude = d + m / 60.

        if not (0. <= longitude <= 180):
            raise ValueError('Longitude format is invalid')

        if ordinal == 'W':
            longitude = -longitude

        return longitude
