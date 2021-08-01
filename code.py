
# FINAL LAB
# Rachel Edelstein - 200317329
# Roni Savitsky - 208919688

import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import nnls
from sklearn.decomposition import PCA


# Read election results from csv file
def read_election_results(year, analysis):
    df_raw = pd.read_csv(os.path.join(DATA_PATH, r'votes per ' + analysis + ' ' + year + '.csv'),
                             encoding='iso-8859-8', index_col='שם ישוב').sort_index()
    if year == '2019b':
        df = df_raw.drop('סמל ועדה', axis=1)  # new column added in Sep 2019
    else:
        df = df_raw

    df = df[df.index != 'מעטפות חיצוניות']
    df_raw = df_raw[df_raw.index != 'מעטפות חיצוניות']
    if analysis == 'city':
        first_col = 5
    else:  # ballot
        first_col = 6
    df = df[df.columns[first_col:]]  # removing "metadata" columns

    return df, df_raw

# Change dataframe to include unique index for each ballot. From Harel Kain
def adapt_df(df, parties, include_no_vote=False, ballot_number_field_name=None):
    df['ballot_id'] = df['סמל ישוב'].astype(str) + '__' + df[ballot_number_field_name].astype(str)
    df_yeshuv = df.index  # new: keep yeshuv
    df = df.set_index('ballot_id')
    eligible_voters = df['בזב']
    total_voters = df['מצביעים']
    df = df[parties]
    df['ישוב'] = df_yeshuv  # new: keep yeshuv
    df = df.reindex(sorted(df.columns), axis=1)

    return df, eligible_voters


# Path to datafiles - change to your directory!
DATA_PATH = "C:/Users/upp3s/Documents/LAB"

# Parties dict
parties_dict = { 'כף': 'עוצמה יהודית', 'ל': 'ישראל ביתנו', 'מחל': 'הליכוד', 'אמת': 'העבודה גשר', 'ג': 'יהדות התורה', 'ודעם': 'הרשימה המשותפת', 'טב': 'ימינה',
                'מרצ': 'המחנה הדמוקרטי', 'פה': 'כחול לבן',  'שס': 'שס'}

parties_dict_2019a = {'אמת': "עבודה", 'ג': "יהדות התורה", 'דעם': "רעם בלד", 'ום': "חדש תעל", 'נ': "ימין חדש", 'טב': "איחוד מפלגות הימין", 'כ': "כולנו",
 'ל': "ישראל ביתנו", 'מחל': "הליכוד", 'מרצ': "מרצ",  'ז': "זהות", 'נר': "גשר", 'פה': "כחול לבן", 'שס': "שס"}

################## Analysis ##################
# Read data for differnt elections
df_april_ballot, df_april_raw_ballot = read_election_results('2019a', 'ballot')
df_sep_ballot, df_sep_raw_ballot = read_election_results('2019b', 'ballot')
df_april, df_april_raw = read_election_results('2019a', 'city')
df_sep, df_sep_raw = read_election_results('2019b', 'city')


# נתונים ספטמבר
df_sep_10 = df_sep[parties_dict.keys()]
df_sep_ballot10, bazab_sep = adapt_df(df_sep_raw_ballot, list(parties_dict.keys()), include_no_vote=False, ballot_number_field_name='קלפי')
yeshuv_sep = df_sep_ballot10['ישוב']
df_sep_ballot10 = df_sep_ballot10.drop('ישוב', axis=1)


q_mat_sep = df_sep_ballot10.div(df_sep_ballot10.sum(axis=1), axis=0) # get voting frequencies
q_mat_sep = q_mat_sep.dropna()

num_voters_sep = df_sep_ballot10['אמת'].sum()


# נתונים אפריל
df_ap_14 = df_april[parties_dict_2019a.keys()]
df_ap_ballot14, bazab_ap = adapt_df(df_april_raw_ballot, list(parties_dict_2019a.keys()), ballot_number_field_name='מספר קלפי')
yeshuv_ap = df_ap_ballot14['ישוב']
df_ap_ballot14 = df_ap_ballot14.drop('ישוב', axis=1)


q_mat_ap = df_ap_ballot14.div(df_ap_ballot14.sum(axis=1), axis=0) # get voting frequencies
q_mat_ap = q_mat_ap.dropna()

num_voters_avoda = df_ap_ballot14['אמת'].sum()
num_voters_gesher = df_ap_ballot14['נר'].sum()

#קלפיות משותפות
shared_cities = df_sep_ballot10.index.intersection(df_ap_ballot14.index)
df_ap_shared = df_ap_ballot14.loc[shared_cities]
df_sep_shared = df_sep_ballot10.loc[shared_cities]

q_mat_ap_shared = df_ap_shared.div(df_ap_shared.sum(axis=1), axis=0) # get voting frequencies
q_mat_ap_shared = q_mat_ap_shared.dropna()

q_mat_sep_shared = df_sep_shared.div(df_sep_shared.sum(axis=1), axis=0) # get voting frequencies
q_mat_sep_shared = q_mat_sep_shared.dropna()

# c.
# Bar plot for all parties with votes above a threshold
def parties_bar(votes1, votes2, head, parties_name):
    width = 0.3
    n = len(votes1)  # number of parties
    fig, ax = plt.subplots()  # plt.subplots()

    city1_bar = ax.bar(np.arange(n), list(votes1), width, color='red')
    city2_bar = ax.bar(np.arange(n)+width, list(votes2), width, color='blue')

    ax.set_ylabel('Votes percent')
    ax.set_xlabel('Parties Names')
    ax.set_title('Votes percent per party 2019 ' + head)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(parties_name, rotation = 90, size = 7)

    if head == 'september':
        plt.text(7- 0.15, 0.055, num_voters_sep, fontsize=7, color="red", fontweight='bold', rotation = 90)
        plt.text(7+ 0.15, 0.055, num_voters_opt_b, fontsize=7, color="blue", fontweight='bold', rotation = 90)
    if head == 'april':
        plt.text(5- 0.15, 0.055, num_voters_avoda, fontsize=7, color="red", fontweight='bold', rotation = 90)
        plt.text(5+ 0.15, 0.055, num_voters_opt_avoda, fontsize=7, color="blue", fontweight='bold', rotation = 90)
        plt.text(13 - 0.15, 0.03, num_voters_gesher, fontsize=7, color="red", fontweight='bold', rotation=90)
        plt.text(13 + 0.15, 0.03, num_voters_opt_gesher, fontsize=7, color="blue", fontweight='bold', rotation=90)


    ax.legend((city1_bar[0], city2_bar[0]), ('הצבעה בפועל'[::-1], 'כל בעלי זכות בחירה'[::-1])) # add a legend

    return fig, ax

#National voting frequency for each party for all holders of suffrage
def theoretical_votes(df, opt_votes):
    correction = df.sum(axis=1).div(opt_votes) # votes per city / Holders of suffrage per city
    new_df = df.iloc[:, 0:len(df.columns)].div(correction, axis = 0) #מנופח
    nation_party_freq = new_df.sum().div(opt_votes.sum()) # votes per party / country-wide holders of suffrage
    return nation_party_freq, new_df



#שכיחות הצבעה אפריל

np.seterr(divide='ignore', invalid='ignore')
theoretical_party_freq_a, df_ap_big = theoretical_votes(df_ap_ballot14, bazab_ap)

num_voters_opt_avoda = round(df_ap_big['אמת'].sum())
num_voters_opt_gesher = round(df_ap_big['נר'].sum())
votes1_a = df_ap_ballot14.sum().div(df_ap_ballot14.sum().sum()).sort_values(ascending=False)  # Percentage of voters in Israel
votes2_a = theoretical_party_freq_a.sort_values(ascending=False)

parties_names_ap = [parties_dict_2019a.get(name)[::-1] for name in votes1_a.keys()]

fig1, ax1 = parties_bar(votes1_a, votes2_a, 'april', parties_names_ap)
fig1.savefig(os.path.join(DATA_PATH, 'Barplot_ballot.png'))  # save the figure
plt.show()

#שכיחות הצבעה ספטמבר

np.seterr(divide='ignore', invalid='ignore')
theoretical_party_freq_b, df_sep_big = theoretical_votes(df_sep_ballot10, bazab_sep)
num_voters_opt_b = round(df_sep_big['אמת'].sum())

votes1_b = df_sep_ballot10.sum().div(df_sep_ballot10.sum().sum()).sort_values(ascending=False)  # Percentage of voters in Israel
votes2_b = theoretical_party_freq_b.sort_values(ascending=False)

parties_names_sep = [parties_dict.get(name)[::-1] for name in votes1_b.keys()]

fig2, ax2 = parties_bar(votes1_b, votes2_b, 'september', parties_names_sep)
fig2.savefig(os.path.join(DATA_PATH, 'Barplot_ballot.png'))  # save the figure
plt.show()


# e.a


# cities names
#city = list(q_mat.index)

# שכיחות ההצבעה לעבודה גשר בכל בקלפי
q_mat_emet = q_mat_sep_shared['אמת'].sort_values()
q_mat_sep_order = q_mat_sep_shared.sort_values(by = 'אמת')

pca = PCA(n_components=2)  # define PCA object
principalComponents = pca.fit_transform(q_mat_sep_order)  # fit model. Compute principal components
X_pca = pca.transform(q_mat_sep_order)  # Perform PCA transformation

fig5, ax5 = plt.subplots(figsize=(10,10))


val = 200 / q_mat_emet[len(q_mat_emet)-1]  # The value that will give size 200 for the largest settlement
sizes = list(val * q_mat_emet)

colors = []
for i in range(len(q_mat_emet)):
    if q_mat_emet.iloc[i] > 0.4:
        colors.append("red")
    elif q_mat_emet.iloc[i] > 0.2 and q_mat_emet.iloc[i] < 0.4:
        colors.append("blue")
    else:
        colors.append("powderblue")

import matplotlib.patches as mpatches

ax5.scatter(X_pca[:,0],X_pca[:,1], s= sizes, c=colors,  alpha=0.45)
city1 = mpatches.Patch(color="red",label="יותר מ%04 מצביעים"[::-1])
city2 = mpatches.Patch(color="blue",label= "בין מ%01 ל%04 מצביעים"[::-1])
all_cities = mpatches.Patch(color="powderblue",label= "פחות מ%01 מצביעים"[::-1])

ax5.legend(handles=[city1,city2,all_cities])


ax5.set_xlabel('PC1', fontsize=10, fontweight="bold")
ax5.set_ylabel('PC2', fontsize=10, fontweight="bold")
ax5.set_title("PCA plot for ballots in 2019 september elections", fontsize=13, fontweight="bold", color='c')

fig5.savefig(os.path.join(DATA_PATH, 'PCA ballots.png'))  # save the figure
plt.show()


# e.b.

q_ap_emet = q_mat_ap_shared['אמת'] + q_mat_ap_shared['נר']
dist = q_mat_sep_shared['אמת'] - q_ap_emet

#top10_max_change = dist.nlargest(10).index
q_mat_sep_shared['dist'] = dist

#q_mat_sep_order2 = q_mat_sep_shared.sort_values(by = 'dist')
#dist_order = q_mat_sep_order2['dist']
#q_mat_sep_order2 = q_mat_sep_order2.drop('dist', axis=1)
#q_mat_sep_shared = q_mat_sep_shared.drop('dist', axis=1)


index_1 = []
index_2 = []

for i in range(len(q_mat_sep_shared)):
    dist_i = q_mat_sep_shared.iloc[i]['dist']
    if dist_i < - 0.1 or dist_i > 0.2: #לא חל שינוי גדול
        index_2.append(q_mat_sep_shared.index[i])
    else:
        #sizes.append(10)
        index_1.append(q_mat_sep_shared.index[i])
q_mat_sep_order2 = q_mat_sep_shared.reindex(index_1 + index_2)

dist_order = q_mat_sep_order2['dist']
q_mat_sep_order2 = q_mat_sep_order2.drop('dist', axis=1)

sizes = [10 for i  in range(len(index_1))]+[70 for j  in range(len(index_2))]


print(sizes)
print(list(dist_order))

pca = PCA(n_components=2)  # define PCA object
principalComponents = pca.fit_transform(q_mat_sep_order2)  # fit model. Compute principal components
X_pca = pca.transform(q_mat_sep_order2)  # Perform PCA transformation

fig6, ax6 = plt.subplots(figsize=(10,10))



colors = []
for i in range(len(dist_order)):
    if dist_order.iloc[i] > 0.2:
        colors.append("red")
    elif dist_order.iloc[i] < -0.1:
        colors.append("blue")
    else:
        colors.append("powderblue")
print(colors)
import matplotlib.patches as mpatches

ax6.scatter(X_pca[:,0],X_pca[:,1], s= sizes, c=colors,  alpha=0.45)
city1 = mpatches.Patch(color="red",label="התחזקה ביותר מ%02"[::-1])
city2 = mpatches.Patch(color="blue",label= "נחלשה ביותר מ%01"[::-1])
all_cities = mpatches.Patch(color="powderblue",label= "כל השאר"[::-1])

ax6.legend(handles=[city1,city2,all_cities])


ax6.set_xlabel('PC1', fontsize=10, fontweight="bold")
ax6.set_ylabel('PC2', fontsize=10, fontweight="bold")
ax6.set_title("PCA plot for ballots in 2019 september elections", fontsize=13, fontweight="bold", color='c')

fig6.savefig(os.path.join(DATA_PATH, 'PCA ballots.png'))  # save the figure
plt.show()


# d


# 1. Intersect election file with demography file
DEM_PATH = "C:/Users/upp3s/Documents/LAB/lab_4"
df_hev = pd.read_table(os.path.join(DEM_PATH, r'HevratiCalcaliYeshuvim.txt'), index_col='רשות מקומית').sort_index()  # , encoding='utf-8')

# Name Correction
def fix_names(df):
    for i in range(len(df.index)-2):
        if 'יי' in df.index[i]:
            df.index.values[i] = df.index.values[i].replace('יי','י')
        if '-' in df.index[i]:
            df.index.values[i] = df.index.values[i].replace('-', ' ')
        if '  ' in df.index[i]:
            df.index.values[i] = df.index.values[i].replace('   ', ' ').replace('  ', ' ')

    return

fix_names(df_sep_10)
fix_names(df_hev)
fix_names(df_ap_14)
fix_names(df_sep_raw)

# pandas intersect command for df_hev, df_sep - find the cities which are in both files
shared_cities = df_hev.index.intersection(df_sep_10.index)
#print("merged cities number:" + str(len(shared_cities)))


# merged data frame
merged_df = pd.concat([df_hev.loc[shared_cities], df_sep_10.loc[shared_cities]],axis=1)
#merged_df_10 = merged_df[merged_df.columns[13:]]

names = ['פה', 'מחל', 'ודעם', 'שס', 'ל', 'ג', 'טב', 'אמת', 'מרצ', 'כף']

#3. Bar Plot votes for each party in a different subplot. Use plt.subplots
# transpose: each row is a party and each column is eshkol

P_dem = np.zeros([10,10])  # P_dem[i][j] is voting frequency for party j at eshkol i


for eshkol in range(10):
    # all cities in current eshkol - finds the index of all cities in each eshkol
    cur_cities = np.where(merged_df['מדד חברתי-'].values.astype('float') == eshkol+1)

    # Compute parties frequencies for cities in current eshkol
    P_dem[eshkol,] = (merged_df.iloc[cur_cities][names].sum())\
        .div(merged_df.iloc[cur_cities][names].sum().sum())

P_dem1 = P_dem.transpose()  # P_dem1[j][i] is voting frequency for party j at eshkol i


# Bar Plot votes for each party in a different subplot.
#fig3,axes3= plt.subplots(2,5,figsize=(40,20))
#for i, ax in enumerate(axes3.flatten()):
fig, ax = plt.subplots()  # plt.subplots()
width = 0.7
n=10
ax.bar(np.arange(n), list(P_dem1[7,]), width)
ax.set_title("העבודה גשר"[::-1], size=15, fontweight='bold', color="red")
ax.set_ylabel("Votes percent", size=9, fontweight='bold',color="blue")
ax.set_xticks(np.arange(n))
ax.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], size=7, fontweight='bold')
ax.set_xlabel("Eshkol", size=9, fontweight='bold', color="blue")
#ax.set_ylim(0, 0.6)
plt.suptitle('Eshkols votes frequency', size=12, fontweight='bold')
plt.subplots_adjust(hspace=0.7, wspace=0.4)

plt.show() # show
#fig3.savefig(os.path.join(DEM_PATH, 'subplots_parties.png'))  # save the figure


shared_cities_ap = df_hev.index.intersection(df_ap_14.index)

# merged data frame
merged_df_ap = pd.concat([df_hev.loc[shared_cities_ap], df_ap_14.loc[shared_cities_ap]],axis=1)
#merged_df_ap_14 = merged_df[merged_df.columns[13:]]

P_dem_ap = np.zeros([10,14])  # P_dem[i][j] is voting frequency for party j at eshkol i
names = parties_dict_2019a.keys()
for eshkol in range(10):
    # all cities in current eshkol - finds the index of all cities in each eshkol
    cur_cities = np.where(merged_df_ap['מדד חברתי-'].values.astype('float') == eshkol+1)
    # Compute parties frequencies for cities in current eshkol
    P_dem_ap[eshkol,] = (merged_df_ap.iloc[cur_cities][names].sum())\
        .div(merged_df_ap.iloc[cur_cities][names].sum().sum())

P_dem1_ap = P_dem_ap.transpose()  # P_dem1[j][i] is voting frequency for party j at eshkol i

# Bar Plot votes for each party in a different subplot.
#fig3,axes3= plt.subplots(2,5,figsize=(40,20))
#for i, ax in enumerate(axes3.flatten()):
fig8, ax8 = plt.subplots(1,2)  # plt.subplots()
width = 0.7
n=10
ax8[0].bar(np.arange(n), list(P_dem1_ap[0,]), width)
ax8[0].set_title("העבודה"[::-1], size=15, fontweight='bold', color="red")
ax8[0].set_ylabel("Votes percent", size=9, fontweight='bold',color="blue")
ax8[0].set_xticks(np.arange(n))
ax8[0].set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], size=7, fontweight='bold')
ax8[0].set_xlabel("Eshkol", size=9, fontweight='bold', color="blue")

ax8[1].bar(np.arange(n), list(P_dem1_ap[11,]), width)
ax8[1].set_title("גשר"[::-1], size=15, fontweight='bold', color="red")
ax8[1].set_ylabel("Votes percent", size=9, fontweight='bold',color="blue")
ax8[1].set_xticks(np.arange(n))
ax8[1].set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], size=7, fontweight='bold')
ax8[1].set_xlabel("Eshkol", size=9, fontweight='bold', color="blue")


plt.suptitle('Eshkols votes frequency', size=12, fontweight='bold')
plt.subplots_adjust(hspace=0.7, wspace=0.4)

plt.show() # show
#fig3.savefig(os.path.join(DEM_PATH, 'subplots_parties.png'))  # save the figure




# מדד ג'יני מול שכיחות הצבעה

# Scatter plot of Gini index and heterogenity
gini_val = df_hev.loc[shared_cities,"מדד ג'יני[2]"]
madad_hev = df_hev.loc[shared_cities,'מדד חברתי-'].values.astype('float')
q_sep_emet = df_sep.div(df_sep.sum(axis=1), axis=0).loc[shared_cities,"אמת"] # get voting frequencies
print(q_sep_emet.sort_values(ascending=False)[188:199])
city_size = df_sep_raw.loc[shared_cities,'בזב']
#print(city_size)
sizes_aray = (city_size.div(city_size.sum())).sort_values(ascending=False)
val = 200 / sizes_aray[0] + 50 # The value that will give size 200 for the largest settlement
city_size = val * (city_size.div(city_size.sum()))


def gini_city_scatter(gini, q_emet , madad, sizes):
    fig, ax = plt.subplots()

    plt.scatter(gini, q_emet, c= madad, s=sizes, alpha=0.4, cmap=plt.cm.get_cmap('jet', 10))  # scatter plot
    plt.ylabel('votes frequencies', fontsize=12, fontweight='bold')
    plt.xlabel('Gini index',fontsize=12, fontweight='bold')
    plt.title('Scatter plot of Gini index and votes frequencies of emet', fontsize=14, color="c", fontweight='bold')

    # Create colorbar
    plt.colorbar(ticks=range(10), label='madad')
    plt.text(0.45, 0.4, 'בית גן'[::-1], fontsize=6, color="red", fontweight='bold')

    '''
    city_ind = [31,66,83,196]
    for i in city_ind:
        plt.text(gini[i] + 0.0075, het[i], het.keys()[i][::-1], fontweight='bold', fontsize=8, color="red",)
    plt.text(gini[42] - 0.017, het[42] + 0.001, het.keys()[42][::-1], fontsize=6, color="red", fontweight='bold')
    plt.text(gini[101] - 0.017, het[101] + 0.001, het.keys()[101][::-1],  fontsize=6, color="red", fontweight='bold')
    plt.text(gini[136] - 0.004, het[136] + 0.002, het.keys()[136][::-1], fontsize=6, color="red", fontweight='bold')
    '''
    plt.show()

    return fig, ax


fig4, ax4 = gini_city_scatter(gini_val, q_sep_emet , madad_hev, city_size)
fig4.savefig(os.path.join(DEM_PATH, 'scatter_gini_val_vs_hetro.png'))  # save the figure

