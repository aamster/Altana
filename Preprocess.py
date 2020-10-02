import numpy as np


class Preprocess:
    def __init__(self):
        self.other_countries = None

    def run(self, df, test=False):
        def country_of_origin(df):
            if test:
                df.loc[df['COUNTRY.OF.ORIGIN'].isin(self.other_countries), 'COUNTRY.OF.ORIGIN'] = 'OTHER'
            else:
                counts = df.groupby('COUNTRY.OF.ORIGIN').size()
                low_count = counts[counts < 100].index
                self.other_countries = low_count

                df.loc[df['COUNTRY.OF.ORIGIN'].isin(low_count), 'COUNTRY.OF.ORIGIN'] = 'OTHER'
            df = df[df['COUNTRY.OF.ORIGIN'].notnull()]
            return df

        def consignee(df):
            df.loc[:, 'CONSIGNEE'] = df['CONSIGNEE'].str.replace('[^\w\s]', '')
            df.loc[(df['CONSIGNEE'].str.contains('WALMART')) | (
                df['CONSIGNEE'].str.contains('WAL MART')), 'CONSIGNEE'] = 'WALMART'
            df.loc[df['CONSIGNEE'] == 'NOT AVAILABLE', 'CONSIGNEE'] = np.nan
            return df

        def container_count(df):
            df = df[df['CONTAINER.COUNT'] > 0]
            return df

        def bill_of_lading(df):
            if test:
                return df
            df = df.drop_duplicates(subset=['BILL.OF.LADING'])
            return df

        def shipper(df):
            df.loc[df['SHIPPER'] == '-NOT AVAILABLE-', 'SHIPPER'] = np.nan
            return df

        df = country_of_origin(df)
        df = consignee(df)
        df = container_count(df)
        df = bill_of_lading(df)
        df = shipper(df)

        df = df.drop(['Unnamed: 0'], axis=1)
        df = self._drop(df)
        return df

    @staticmethod
    def _drop(df, drop_text_fields=True):
        missing = ['SHIPPER.ADDRESS', 'CONSIGNEE', 'CONSIGNEE.ADDRESS',
                   'DISTRIBUTION.PORT', 'CARRIER.STATE', 'CARRIER.ZIP', 'SHIPPER']
        drop = missing + ['QUANTITY', 'MEASUREMENT', 'WEIGHT..KG.',
                                'BILL.OF.LADING', 'CARRIER.NAME']
        if drop_text_fields:
            drop += ['PRODUCT.DETAILS', 'MARKS.AND.NUMBERS']
        df = df.drop(drop, axis=1)
        return df