from sklearn.base import  BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, Imputer, StandardScaler
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import check_array
from scipy import sparse


def convert_missing_values(data,lookup_table):
    
    df = data.copy()
    
    for field, encoding in zip(lookup_table['attribute'],lookup_table['missing_or_unknown']):
    
        print('Replacing: {}'.format(field))
    

        if encoding in ['[XX]','[-1,XX]']:
            replace_values = list(map(str,ast.literal_eval(encoding.replace('XX',"'XX'"))))
            
        elif encoding == '[-1,X]':
             replace_values = list(map(str,ast.literal_eval(encoding.replace('X',"'X'"))))
                
        else:
            replace_values = ast.literal_eval(encoding)
                                      
        
        if len(replace_values) > 0:
            df[field].replace(to_replace=replace_values, value=np.nan,inplace=True)
        
    return df

def plot_comparisons(data_set,columns):

    '''
    Plot comparisons between customers with missing values and those without

    Inputs:
        data_set: The dataset on which comparisons needs to be done
        columns: The features that need to be compared
    '''

    number_fields = len(columns)

    ncol=2
    nrow = int(np.ceil(number_fields/ncol))
    graph_width = 20
    graph_height = 5*nrow
    plt.close()
    plt.figure(figsize=(graph_width,graph_height))

    for i in range(number_fields):
        ax = plt.subplot(nrow,ncol,i+1)
        ax = sns.countplot(x=columns[i], data=data_set,hue='Has_missing')
        ax.legend('')

    plt.legend(['No','Yes'],loc='upper left', bbox_to_anchor=(ncol-1, nrow),title='Missing values')
    plt.show()


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features as a numeric array.
    The input to this transformer should be a matrix of integers or strings,
    denoting the values taken on by categorical (discrete) features.
    The features can be encoded using a one-hot aka one-of-K scheme
    (``encoding='onehot'``, the default) or converted to ordinal integers
    (``encoding='ordinal'``).
    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.
    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.
    Parameters
    ----------
    encoding : str, 'onehot', 'onehot-dense' or 'ordinal'
        The type of encoding to use (default is 'onehot'):
        - 'onehot': encode the features using a one-hot aka one-of-K scheme
          (or also called 'dummy' encoding). This creates a binary column for
          each category and returns a sparse matrix.
        - 'onehot-dense': the same as 'onehot' but returns a dense array
          instead of a sparse matrix.
        - 'ordinal': encode the features as ordinal integers. This results in
          a single column of integers (0 to n_categories - 1) per feature.
    categories : 'auto' or a list of lists/arrays of values.
        Categories (unique values) per feature:
        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories are sorted before encoding the data
          (used categories can be found in the ``categories_`` attribute).
    dtype : number type, default np.float64
        Desired dtype of output.
    handle_unknown : 'error' (default) or 'ignore'
        Whether to raise an error or ignore if a unknown categorical feature is
        present during transform (default is to raise). When this is parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the resulting one-hot encoded columns for this feature
        will be all zeros.
        Ignoring unknown categories is not supported for
        ``encoding='ordinal'``.
    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting. When
        categories were specified manually, this holds the sorted categories
        (in order corresponding with output of `transform`).
    Examples
    --------
    Given a dataset with three features and two samples, we let the encoder
    find the maximum value per feature and transform the data to a binary
    one-hot encoding.
    >>> from sklearn.preprocessing import CategoricalEncoder
    >>> enc = CategoricalEncoder(handle_unknown='ignore')
    >>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
    ... # doctest: +ELLIPSIS
    CategoricalEncoder(categories='auto', dtype=<... 'numpy.float64'>,
              encoding='onehot', handle_unknown='ignore')
    >>> enc.transform([[0, 1, 1], [1, 0, 4]]).toarray()
    array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]])
    See also
    --------
    sklearn.preprocessing.OneHotEncoder : performs a one-hot encoding of
      integer ordinal features. The ``OneHotEncoder assumes`` that input
      features take on values in the range ``[0, max(feature)]`` instead of
      using the unique values.
    sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot
      encoding of dictionary items or strings.
    """

    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """Fit the CategoricalEncoder to X.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_feature]
            The data to determine the categories of each feature.
        Returns
        -------
        self
        """

        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        """Transform X using one-hot encoding.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.
        Returns
        -------
        X_out : sparse matrix or a 2-d array
            Transformed input.
        """
        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out

def clean_data(data,encoder,variables_to_be_encoded,drop_variables , missing_threshold=0.2):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data

    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """

    # Put in code here to execute all main cleaning steps:
    # convert missing value codes into NaNs, ...
  
    df = data.copy()
    feat_info = pd.read_csv('AZDIAS_Feature_Summary.csv',delimiter=';')

    print('Encoding Variables')

    for field, encoding in zip(feat_info['attribute'],feat_info['missing_or_unknown']):


        if encoding in ['[XX]','[-1,XX]']:
            replace_values = list(map(str,ast.literal_eval(encoding.replace('XX',"'XX'"))))

        elif encoding == '[-1,X]':
            replace_values = list(map(str,ast.literal_eval(encoding.replace('X',"'X'"))))

        else:
            replace_values = ast.literal_eval(encoding)

        if len(replace_values) > 0:
            df[field].replace(to_replace=replace_values, value=np.nan,inplace=True)

    # remove selected columns and rows, ...

    print('Removing missing values')

    df.drop(['AGER_TYP', 'GEBURTSJAHR', 'TITEL_KZ', 'ALTER_HH', 'KK_KUNDENTYP','KBA05_BAUMAX'],axis=1,inplace=True)

    number_cols = df.shape[1]
    perc_missing_row = df.isnull().sum(axis=1)/number_cols*100
    rows_to_drop = perc_missing_row[perc_missing_row > missing_threshold].index

    df = df.drop(rows_to_drop,axis=0)


    # select, re-encode, and engineer column values.

    print('Creating new variables')

    df['DECADE'] = df['PRAEGENDE_JUGENDJAHRE'].apply(return_decade).astype(float)
    df['MAINSTREAM'] = df['PRAEGENDE_JUGENDJAHRE'].apply(return_movement).astype(float)
    df['WEALTH'] = df['CAMEO_INTL_2015'].astype(float)  // 10
    df['LIFESTAGE'] = df['CAMEO_INTL_2015'].astype(float)  % 10


    print('One-hot encoding variables')

    categorical_one_hot = encoder.transform(df[variables_to_be_encoded].astype(str))

    dummy_column_names = []
    for col, values in zip(variables_to_be_encoded,encoder.categories_):
        dummy_column_names += [(col+"_"+str(i)) for i in values]


    dummy_df = pd.DataFrame(categorical_one_hot,columns=dummy_column_names,index=df.index)

    # Return the cleaned dataframe.

    df = pd.concat([df.drop(drop_variables,axis=1),dummy_df],axis=1)

    return df


def plot_variance(prin_comp,print_cumulative=False, min_percentage_explained=0.8):
    
    '''
    Plot the variance or cumulative variance ratio for each principal component
    Inputs:
    
        prin_comp: A prinicpal component fitted object
        print_cumulative: Whether to print the cumulative variance ratio
        min_percentage_explained: Where to draw the reference line
        
    '''
    
    var_ratio_explained = prin_comp.explained_variance_ratio_
    variances = prin_comp.explained_variance_
    cum_var_ratio_explained = var_ratio_explained.cumsum()
    
    number_components = len(cum_var_ratio_explained)
    
    if print_cumulative:
        
        number_prin_greater_80 = np.min(np.where(cum_var_ratio_explained >= min_percentage_explained)) +1
        plt.rc('font', size=15)
        plt.figure(figsize=(20,10))
        plt.plot(range(1,number_components+1),cum_var_ratio_explained,lw=4)
        plt.ylabel('Cumulative proportion of variance explained')
        plt.xlabel('Number of components')
        plt.title('Cumulative proportion of variance explained by \nnumber of principal components')
        plt.xticks(range(0,number_components+1,10))
        plt.plot([number_prin_greater_80, number_prin_greater_80], [0, min_percentage_explained], 'r--', lw=2)
        plt.plot([0, number_prin_greater_80], [min_percentage_explained, min_percentage_explained], 'r--', lw=2)
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        plt.show()
          
    else:
        
        plt.rc('font', size=15)
        plt.figure(figsize=(20,10))
        plt.plot(range(1,number_components+1),variances,lw=4)
        plt.ylabel('Variance')
        plt.xlabel('Number of components')
        plt.title('Variance explained by \nnumber of principal components')
        plt.xticks(range(0,number_components+1,10))
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        plt.show()

def mapping_principal_components(pca,input_columns,principal_component=1):

    '''
    Map weights for the k-th prinicpal component to the corresponding feature name

    Inputs:
        pca: A fitted PCA object
        input columns: The names of the features that were included in the PCA
        principal_component: The principal component that needs to be mapped
    '''

    loading=pca.components_[principal_component-1].tolist()
    df = pd.DataFrame({'Feature': input_columns, 'Loading': loading})

    return df.sort_values(by='Loading',ascending=False)

def scale_impute_colums(data,means,stdevs):

    '''
    Imputes missing values with supplied values and then scales columns to
    have unit variances and 0 means

    Inputs:
        data: A (panda) dataframe than needs to be imputed and scaled
        means: A list or series containing the means of the fields
        stdevs: A list or series containing the standard deviations of the fields 
    '''


    imputed_data = data.fillna(value=means)
    rescaled_data = (imputed_data - means) / stdevs

    return rescaled_data

def return_decade(field):
    
    if field <= 2:
        return 40
    elif field <= 4:
        return 50
    elif field <= 7:
        return 60
    elif field <= 9:
        return 70
    elif field <= 13:
        return 80
    elif field <= 15:
        return 90
    else:
        return field
    
def return_movement(field):
    
    if field in [1,3,5,8,10,12,14]:
        return 1
    elif field in [2,4,6,7,9,11,13,15]:
        return 0
    else:
        return field
