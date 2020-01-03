import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(limit_stars=True):
    # Load HYG star database
    hyg = pd.read_csv('../data/raw/hygfull.csv')
    if limit_stars:
        hyg = hyg[hyg.Mag <= 6]
    # load asterisms of constellation
    asterisms = pd.read_csv('../data/processed/asterisms.csv')
    asterisms['stars'] = asterisms.stars.apply(lambda x: eval(x))

    return hyg, asterisms

hyg, asterisms = load_data(False)

def hip_to_radec(hip, hyg=hyg):
    res = hyg[hyg.Hip == hip][['RA','Dec']]
    # assert res.shape[0] == 1
    RA, Dec =  res.values[0].tolist()
    return (RA, Dec)

def plot_asterism(constellation_code, asterisms = asterisms, hyg = hyg):
    df = asterisms[asterisms.constellation == constellation_code]
    asterism = df.stars.values[0]
    coords = [hip_to_radec(int(hip)) for hip in asterism]
    x, y = map(np.array, zip(*coords))

    for i in range(0, len(x), 2):
        plt.plot(x[i:i+2], y[i:i+2], 'bo-')

    plt.annotate(df.constellation.values[0], (x.mean(), y.mean()))

def plot_asterisms(constellation_codes, asterisms = asterisms, hyg = hyg):
    assert type(constellation_codes) == list
    for code in constellation_codes:
        plot_asterism(code, asterisms, hyg)


plot_asterism('Aql')

df = asterisms[asterisms.constellation == 'Aql']
asterism = df.stars.values[0]
coords = [hip_to_radec(hip) for hip in asterism]




df['Constl'] = df.BayerFlamsteed.apply(lambda x: str(x).split(' ')[-1])
df[df['Constl'] != '']
locals()['df']


df

hip_to_radec(145)
asterism = ['85927', '86670', '86670', '87073', '87073', '86228', '86228', '84143', '84143', '82671', '82671', '82514', '82514', '82396', '82396', '81266', '81266', '80763', '80763', '78401', '80763', '78265', '80763', '78820']
coords = [hip_to_radec(int(coords)) for coords in asterism]

x, y = map(np.array, zip(*coords))

df[df.Hip == 85927]

for i in range(0, len(x), 2):
    plt.plot(x[i:i+2], y[i:i+2], 'ro-')

plt.show()

hyg


plt.figure(figsize=(20,20))
for constl in asterisms.constellation:
    print(constl)
    plot_asterism(constl)
plt.show


asterisms.constellation.unique()
plot_asterism('Leo')
tmp = asterisms[asterisms.constellation == 'Cet'].stars.values[0]

[int(star) in hyg.Hip for star in tmp]
plot_asterism('Scl')


plot_asterisms(['Scl', 'PsA'])
plt.show()

tmp = asterisms.query('constellation == "Scl"').stars.values[0]
tmp = list(set(tmp))
hyg[hyg.Hip.isin(tmp)]
