from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd


class FeatureEngineering:
    def __init__(self):
        self.count_encode = None
        self.one_hot = None
        self.count_encodings = {}
        self.one_hot_enc = {}
        self.scaler = None

    def run(self, df, test=False):
        df['arrival_month'] = df['ARRIVAL.DATE'].dt.month
        df = df.drop('ARRIVAL.DATE', axis=1)
        df = self._categorical_encode(df=df, test=test)
        df = self._std_scaler(df=df, test=test)
        return df

    def _categorical_encode(self, df, test=False):
        df_ = df.select_dtypes(include='object')

        df_ = df_.fillna('missing')

        if test:
            for col in self.count_encode:
                df_[col] = self._count_encode(col=df_[col], test=True)
        else:
            nunique = df_.nunique()

            count_encode = nunique[nunique > 100].index
            self.count_encode = count_encode

            for col in count_encode:
                df_[col] = self._count_encode(col=df_[col])

            one_hot = [c for c in df_ if c not in count_encode]
            self.one_hot = one_hot

        one_hots = []
        for col in self.one_hot:
            res = self._one_hot_encode(df=df_, col=col, test=test)
            one_hots.append(pd.DataFrame(res, columns=self.one_hot_enc[col].categories_, index=df_.index))

        one_hots = pd.concat(one_hots, axis=1)
        cat = pd.concat([df_[self.count_encode], one_hots], axis=1)

        df = df.drop(self.count_encode.tolist() + self.one_hot, axis=1)
        df = pd.concat([df, cat], axis=1)

        return df

    def _std_scaler(self, df, test=False):
        if test:
            df = self.scaler.transform(df)
        else:
            self.scaler = StandardScaler()
            df = self.scaler.fit_transform(df)
        return df

    def _count_encode(self, col, test=False):
        if test:
            enc = self.count_encodings[col.name]
            res = col.map(enc)
            res = res.fillna(enc.max() + 1)
            return res
        else:
            value_counts = col.value_counts()
            self.count_encodings[col.name] = value_counts
            return col.map(value_counts)

    def _one_hot_encode(self, df, col, test=False):
        if test:
            res = self.one_hot_enc[col].transform(df[[col]])
        else:
            self.one_hot_enc[col] = OneHotEncoder(handle_unknown='ignore', sparse=False)
            res = self.one_hot_enc[col].fit_transform(df[[col]])
        return res

def test():
    import pandas as pd
    import numpy as np
    from Preprocess import Preprocess
    train = pd.read_csv('~/Downloads/ds-project-train.csv', dtype={'SHIPPER.ADDRESS': np.str, 'ZIPCODE': np.str},
                     parse_dates=['ARRIVAL.DATE'])
    test = pd.read_csv('~/Downloads/ds-project-test.csv', dtype={'SHIPPER.ADDRESS': np.str, 'ZIPCODE': np.str},
                     parse_dates=['ARRIVAL.DATE'])
    p = Preprocess()
    X_train = p.run(df=train)
    X_test = p.run(df=test, test=True)

    y_train = X_train['COUNTRY.OF.ORIGIN']
    X_train = X_train.drop(['COUNTRY.OF.ORIGIN'], axis=1)

    y_test = X_test['COUNTRY.OF.ORIGIN']
    X_test = X_test.drop(['COUNTRY.OF.ORIGIN'], axis=1)


    fe = FeatureEngineering()
    X_train = fe.run(df=X_train)
    X_test = fe.run(df=X_test, test=True)

    print('!')
if __name__ == '__main__':
    test()