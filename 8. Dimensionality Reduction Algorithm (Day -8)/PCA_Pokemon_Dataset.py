import pandas as pd
from sklearn.decomposition import PCA
import math
import matplotlib.pyplot as plt

pokemon = pd.read_csv('pokemon.csv')

print(pokemon.head())

# Just take these features of interest
df = pokemon[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]

print(df.describe())

pca = PCA(n_components=2, svd_solver='full')
pca.fit(df)

T = pca.transform(df)

# Started with 6 dimensions
print(df.shape)
# Left with 2 principle components
print(T.shape)

print(df.head())

print(pca.explained_variance_ratio_)

components = pd.DataFrame(pca.components_, columns = df.columns, index=[1, 2])

print(components)

#We can do some mathematics to find out which are the most important features:

def get_important_features(transformed_features, components_, columns):
    """
    This function will return the most "important" 
    features so we can determine which have the most
    effect on multi-dimensional scaling
    """
    num_columns = len(columns)

    # Scale the principal components by the max value in
    # the transformed set belonging to that component
    xvector = components_[0] * max(transformed_features[:,0])
    yvector = components_[1] * max(transformed_features[:,1])

    # Sort each column by it's length. These are your *original*
    # columns, not the principal components.
    important_features = { columns[i] : math.sqrt(xvector[i]**2 + yvector[i]**2) for i in range(num_columns) }
    important_features = sorted(zip(important_features.values(), important_features.keys()), reverse=True)
    print("Features by importance:\n", important_features)

get_important_features(T, pca.components_, df.columns.values)

#By plotting these lengths, we can see this visually:

def draw_vectors(transformed_features, components_, columns):
    """
    This funtion will project your *original* features
    onto your principal component feature-space, so that you can
    visualize how "important" each one was in the
    multi-dimensional scaling
    """

    num_columns = len(columns)

    # Scale the principal components by the max value in
    # the transformed set belonging to that component
    xvector = components_[0] * max(transformed_features[:,0])
    yvector = components_[1] * max(transformed_features[:,1])

    ax = plt.axes()

    for i in range(num_columns):
    # Use an arrow to project each original feature as a
    # labeled vector on your principal component axes
        plt.arrow(0, 0, xvector[i], yvector[i], color='b', width=0.0005, head_width=0.02, alpha=0.75)
        plt.text(xvector[i]*1.2, yvector[i]*1.2, list(columns)[i], color='b', alpha=0.75)

    return ax

ax = draw_vectors(T, pca.components_, df.columns.values)
T_df = pd.DataFrame(T)
T_df.columns = ['component1', 'component2']

T_df['color'] = 'y'
T_df.loc[T_df['component1'] > 125, 'color'] = 'g'
T_df.loc[T_df['component2'] > 125, 'color'] = 'r'

plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')
plt.scatter(T_df['component1'], T_df['component2'], color=T_df['color'], alpha=0.5)
plt.show()


# High Attack, High Sp. Atk, all of these pokemon are legendary
print(pokemon.loc[T_df[T_df['color'] == 'g'].index])

# High Defense, Low Speed
print(pokemon.loc[T_df[T_df['color'] == 'r'].index])