get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import sys
from collections import OrderedDict

sys.path.append('../../tools/music-processing-experiments')

from time_intervals import block_labels
from analysis import split_block_times


# let's generate some segments represented by their start time
def segments_from_events(event_times, labels=None):
    """Create a DataFrame with adjacent segments from a list of border events."""
    segments = pd.DataFrame({'start': event_times[:-1], 'end': event_times[1:]})
    segments['duration'] = segments['end'] - segments['start']
    segments = segments[['start', 'end', 'duration']]
    if labels != None:
        assert(len(event_times) == len(labels) + 1)
        segments['label'] = labels
    return segments

def events_from_segments(segments):
    """Create a list of border events DataFrame with adjacent segments."""
    return np.hstack([segments['start'], segments['end'].iloc[-1]])

def generate_segments(size, seed=0):
    np.random.seed(seed)
    event_times = np.random.normal(loc=1, scale=0.25, size=size+1).cumsum()
    event_times = event_times - event_times[0]
    return segments_from_events(event_times, np.random.randint(4, size=size))


segments = generate_segments(20)
segments.head()


events_from_segments(segments)


def plot_segments(segments, seed=42):
    size = len(segments)
    np.random.seed(seed)
    if 'label' not in segments.columns:
        colors = np.random.permutation(size) / size
    else:
        labels = segments['label']
        unique_labels = sorted(set(segments['label']))
        color_by_label = dict([(l, i) for (i, l) in enumerate(unique_labels)])
        norm_factor = 1.0 / len(unique_labels)
        colors = labels.apply(lambda l: color_by_label[l] * norm_factor)
    plt.figure(figsize=(20,5))
    plt.bar(segments['start'], np.ones(size), width=segments['duration'], color=cm.jet(colors), alpha=0.5)
    plt.xlim(0, segments['end'].iloc[-1])
    plt.xlabel('time')
    plt.yticks([]);


plot_segments(segments)


def make_blocks(total_duration, block_duration):
    return segments_from_events(np.arange(0, total_duration, block_duration))


total_duration = segments.iloc[-1]['end']
print('total duration:', total_duration)
blocks = make_blocks(total_duration, 0.25)
print('number of blocks:', len(blocks))
blocks.head()


plot_segments(blocks)


# # Representation of events
# 

# - just a list of start times + label, sentinel at the end
#   - (+) easy to merge
#   - (-) needs a sentinel
# - list of segments with start & end times and label
#   - (+) provided eg. in the Isophonics dataset
#   - (+) easy to reason about and visualize
#   - (-) harder to merge
# 

class Events():
    def __init__(self, start_times, labels):
        """last item must be sentinel with no label"""
        assert(len(labels) >= len(start_times) - 1)
        if len(labels) < len(start_times):
            labels = labels.append(pd.Series([np.nan]))
        self._df = pd.DataFrame({'start': start_times, 'label': labels}, columns=['start', 'label'])
    def df(self):
        return self._df
    
class Segments():
    def __init__(self, start_times, labels):
        """last item must be sentinel with NaN label"""
        self._df = segments_from_events(start_times, labels)

    def df(self):
        return self._df
    
    def join(self, other):
        sentinel_value = '_END_'
        def add_sentinel(df):
            last_event = df[-1:]
            return df.append(pd.DataFrame({
                'start': last_event['end'],
                'end': last_event['end'],
                'duration': 0.0,
                'label': sentinel_value
            }, columns=last_event.columns))
        def remove_sentinel(df, cols):
            for col in cols:
                df[col] = df[col].apply(lambda v: np.nan if v == sentinel_value else v)
        self_df = add_sentinel(self.df())[['start', 'label']].set_index('start')
        other_df = add_sentinel(other.df())[['start', 'label']].set_index('start')
        joined_df = self_df.join(other_df, lsuffix='_left', rsuffix='_right', how='outer')
        joined_df.fillna(method='ffill', inplace=True)
        remove_sentinel(joined_df, ['label_right', 'label_left'])
        joined_df['label_equals'] = joined_df['label_left'] == joined_df['label_right']
        joined_df.reset_index(inplace=True)
        joined_df['end'] = joined_df['start'].shift(-1)
        joined_df['duration'] = joined_df['end'] - joined_df['start']
        joined_df = joined_df[:-1]
        return joined_df #Segments(joined_df['start'], joined_df['label'])


annotations = Segments(np.array([0, 1, 2, 3, 3.5, 4]), ['A','B','A','C','A'])
annotations.df()


plot_segments(annotations.df())


estimations = Segments(np.array([0, 0.9, 1.8, 2.5, 3.1, 3.4, 4.5]), ['A','B','A','B','C','A'])
estimations.df()


plot_segments(estimations.df())


def join_segments(df1, df2):
    """Joins two dataframes with segments into a single one (ignoring labels)"""
    np.hstack(events_from_segments(df1), events_from_segments(df1))


events = np.hstack([events_from_segments(annotations), events_from_segments(estimations)])
events.sort()
events = np.unique(events)
events


merged = segments_from_events(events)
merged


plot_segments(merged)


merged_df = annotations.join(estimations)
merged_df


# # [Weighted] Chord Symbol Recall
# 
# ```
# CSR = D_equals / D_annot
# ```
# 
# - `CSR` - chord symbol recall (for a single song)
# - `D_equals` - total duration of segments where annotation equals estimation
# - `D_annot` - total duration of annotated segments
# 
# ```
# WSCR = sum(D_annot[i] * CSR[i])
# ```
# 
# - `WSCR` - weighted CSR - average of CSR across songs weighted by song durations
# 

def chord_symbol_recall(pred_segments, true_segments):
    merged_df = pred_segments.join(true_segments)
    return merged_df[merged_df['label_equals']]['duration'].sum() / merged_df['duration'].sum()


chord_symbol_recall(estimations, annotations)


# # Blocks to segments
# 
# - merge adjacent blocks with same labels into segments
# 




get_ipython().magic('pylab inline')

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import glob

pylab.rcParams['figure.figsize'] = (16, 12)


data_dir = 'data/beatles/chordlab/The_Beatles/'


chord_files = glob.glob(data_dir + '*/*.lab.pcs.tsv')


print('total number of songs', len(chord_files))
chord_files[:5]


def read_chord_file(path):
    return pd.read_csv(path, sep='\t')

def add_track_id(df, track_id):
    df['track_id'] = track_id
    return df

def track_title(path):
    return '/'.join(path.split('/')[-2:]).replace('.lab.pcs.tsv', '')

def read_key_file(path):
    return pd.read_csv(path, sep='\t', header=None, names=['start', 'end', 'silence', 'key_label'])


selected_files = chord_files
all_chords = pd.concat(add_track_id(read_chord_file(file), track_id) for (track_id, file) in enumerate(selected_files))

all_chords['duration'] = all_chords['end'] - all_chords['start']

nonsilent_chords = all_chords[all_chords['label'] != 'N']

print('total number of chord segments', len(all_chords))


key_files = glob.glob('data/beatles/keylab/The_Beatles/*/*.lab')
len(key_files)


all_keys = pd.concat(add_track_id(read_key_file(file), track_id) for (track_id, file) in enumerate(key_files))


print('all key segments:', len(all_keys))
print('non-silence key segments:', len(all_keys['key_label'].dropna()))


all_keys['key_label'].value_counts()


all_keys['key_label'].map(lambda label: label.replace(':.*', ''))


pcs_columns = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']


def find_track(name):
    return [i for (i, path) in enumerate(chord_files) if name in path]


def draw_track(track_id):
    print(track_title(chord_files[track_id]))
    track = all_chords[all_tracks['track_id'] == track_id]
    matshow(track[pcs_columns].T)
    grid(False)
    gca().set_yticks(np.arange(12))
    gca().set_yticklabels(pcs_columns)


# Example time line of chords represented as binary pitch class sets in a single song:
# 

draw_track(find_track('Yesterday')[0])


# Distribution of pitch classes accross all songs:
# 

pc_histogram = pd.DataFrame({'pitch_class': pcs_columns, 'relative_count': nonsilent_chords[pcs_columns].mean()})
stem(pc_histogram['relative_count'])
gca().set_xticks(np.arange(12))
gca().set_xticklabels(pcs_columns);


pc_histogram.sort('relative_count', ascending=False, inplace=True)


plot(pc_histogram['relative_count'],'o:')
gca().set_xticks(np.arange(12))
gca().set_xticklabels(pc_histogram['pitch_class']);
ylim(0, 1);
xlim(-.1, 11.1);


# Observation: Five most used pitch classes in Beates songs are A, E, D, B and G.
# 

chord_histogram = all_chords['label'].value_counts()


chord_histogram


print('number of unique chords (including silence):', len(chord_histogram))


# Distribution of chords usage:
# 

plot(chord_histogram);


# Distribution of chord root tones (ie. chord stripped of their quality) - excluding silence:
# 

chord_root_histogram = nonsilent_chords['root'].value_counts()
# convert index from integers to symbolic names
chord_root_histogram.index = pd.Series(pcs_columns)[chord_root_histogram.index].values
chord_root_histogram


# Still [A,D,G,E,C] is the set of most favorite pitch classes.
# 

#all_chords[pcs_columns + ['track_id']]


all_chords


# Distribution of chord segment duration:
# 

duration = all_chords['duration']
duration.hist(bins=100);


sns.distplot(duration[duration < 10], bins=100)
xlabel('duration (sec)');


# # Let's do some machine learning :)
# 
# ## Task: predict track id (song) - based on chords and segment duration
# 
# ### Start with a independent data points (no context) and logistic regression
# 

# First prepare the data into training, validation and test set.
# 

X = all_chords[['duration'] + pcs_columns].astype(np.float32)
y = all_chords['track_id'].astype(np.int32)


from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

len(X_train), len(X_valid), len(X_test)


from sklearn.linear_model import LogisticRegression


lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)


from sklearn.metrics import classification_report, confusion_matrix

y_pred_lr = lr_model.predict(X_valid)

lr_model.score(X_valid, y_valid)


print(classification_report(y_valid, y_pred_lr))


matshow(confusion_matrix(y_valid, y_pred_lr), cmap=cm.Spectral_r)
colorbar();


import theanets
import climate # some utilities for command line interfaces
climate.enable_default_logging()

exp = theanets.Experiment(
    theanets.Classifier,
    layers=(13, 50, 180),
    hidden_l1=0.1)

exp.train(
    (X_train, y_train),
    (X_valid, y_valid),
    optimize='rmsprop',
    learning_rate=0.01,
    momentum=0.5)


y_pred_nn = exp.network.classify(X_valid)
y_pred_nn


print(classification_report(y_valid, y_pred_nn))


matshow(confusion_matrix(y_valid, y_pred_nn), cmap=cm.Spectral_r)
colorbar();


# # TODO
# 
# - better feature engineering:
#   - scale the features
#   - use context, eg. blocks of chords instead of just a single chord segment
# - use a deep neural network, eg. via theanets to learn better low-dimensional features
# - use an RNN, that would automatically handle the context
# 







# # Segments to frames
# 
# The goal is to map a sequence of variable-length segments to fixed-length frames.
# 
# Input format:
# 
# Chords in the annotated dataset are represented in segments. The song is a sequence of segments. Each segment has a start, end time and value (in this case chord label). Segments have to be contuguous, non-overlapping and non-empty. 
# The song starts and time 0.0.
# 
# Output format:
# 
# We'd like to use features like spectrogram, chomagrams, etc. When a digital audio recording is analyzed with such spectral methods, it is split into fixed-size frames. Each frame represents also a time interval with start and end time. Since the signal in each frame is typically windowed and the frames are overlapping we can describe the frame just by the time of its center.
# 
# Since we'd like to use the chord annotations as labels for the features we have to map them to frames.
# 
# This can be done quite easily. In pseuodocode:
# 
# ```
# for each frame:
#     take its center time
#     find the segment containing this time
#     assign that segment's value as the frame label
# ```
# 

get_ipython().magic('pylab inline')
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
pylab.rcParams['figure.figsize'] = (12, 6)


segments = pd.DataFrame.from_records([
    (0, 2.4, 'C'),
    (2.4, 2.8, 'F'),
    (2.8, 4.7, 'G'),
    (4.7, 5.2, 'C'),
], columns=['start','end','label'])
segments


segment_count = len(segments)
total_duration = segments['end'].iloc[-1]
frame_duration = 1.0
hop_duration = 0.5


def time_intervals(segments):
    return [(v['start'], v['end']) for (k,v) in segments[['start', 'end']].iterrows()]

def plot_segments(time_intervals):
    ax = plt.gca()
    for (i, (s, e)) in enumerate(time_intervals):
        j = (i / 5) % 1
        yshift = 0.1 * (abs(j - 0.5) - 0.5)
        ax.add_patch(Rectangle(
                (s, yshift), e-s, yshift + 1, alpha=0.5, linewidth=2,
                edgecolor=(1,1,1), facecolor=plt.cm.jet(j)))
    pad = 0.1
    xlim(0 - pad, total_duration + pad)
    ylim(0 - pad, 1 + pad)


plot_segments(time_intervals(segments))


def frame_count(total_duration, frame_duration, hop_duration):
    return math.ceil((max(total_duration, frame_duration) - frame_duration) / hop_duration + 1)

frame_count(total_duration, frame_duration, hop_duration)


def frames(total_duration, frame_duration, hop_duration):
    count = frame_count(total_duration, frame_duration, hop_duration)
    return [(i * hop_duration, i * hop_duration + frame_duration) for i in range(count)]

def frame_centers(total_duration, frame_duration, hop_duration):
    count = frame_count(total_duration, frame_duration, hop_duration)
    return [(0.5  * frame_duration+ i * hop_duration) for i in range(count)]


f_centers = frame_centers(total_duration, frame_duration, hop_duration)
f_centers


f = frames(total_duration, frame_duration, hop_duration)
f


plot_segments(f)


def label_at_time(time, segments):
    labels = segments[(segments['start'] <= time) & (segments['end'] >= time)]['label']
    if len(labels) >= 0:
        return labels.iloc[0]


[label_at_time(t, segments) for t in f_centers]





# # Time map of chord events
# 
# Goal: compute and visualize a time map of chord events.
# 
# This provides an overview of timing of events without the need of zooming the plot.
# 
# [Time Maps: Visualizing Discrete Events Across Many Timescales](https://districtdatalabs.silvrback.com/time-maps-visualizing-discrete-events-across-many-timescales) by Max Watson.
# 
# ## Dataset
# 
# Reference Annotations: The Beatles
# 
# - http://isophonics.net/content/reference-annotations-beatles
# 
# ### Format
# - TSV file, each line describes a single segment with a chord
# - columns: `start_time end_time mirex_chord_label`
# - example: `2.9632 6.1260 G:sus4(b7)`
# - time is in float seconds
# - label syntax: https://code.soundsoftware.ac.uk/attachments/download/330/chris_harte_phd_thesis.pdf
# 
# ## Output:
# 
# - time map on:
#   - the whole dataset
#   - each song
#   - blocks of a single song (how the distribution changes)
# - plot type/scale:
#   - scatter plot, 2D histogram plot
#   - linear scale
# 

get_ipython().magic('pylab inline')

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

pylab.rcParams['figure.figsize'] = (16, 12)


data_dir = 'data/beatles/chordlab/The_Beatles/'


file = "03_-_A_Hard_Day's_Night/03_-_If_I_Fell.lab"
def read_chord_file(path):
    return pd.read_csv(path, sep=' ', header=None, names=['start','end','chord'])
df = read_chord_file(data_dir + file)


df.head()


plot(df['start'], 'g.', label='start')
plot(df['end'], 'r.', label='end')
legend(loc='center right');


print('event count:', len(df))
print('total time:', df['end'].iloc[-1])
print('last start event time:', df['start'].iloc[-1])


df['duration'] = df['end'] - df['start']
df['duration'].describe()


plot(df['duration'], '.')
xlabel('segment index')
ylabel('duration');


sns.distplot(df['duration'], axlabel='duration (sec)', rug=True, kde=False, bins=20)
title('Duration of chord segments');


# Time map is just a scatter plot of time to previous event vs. time to next event. Let's compute the differences.
# 

def add_differences(df, col='start'):
    df['prev'] = df[col].diff(1)
    df['next'] = -df[col].diff(-1)
    return df
df_diff = add_differences(df).dropna()


df_diff.head()


sns.distplot(df_diff['prev'], label='time to previous', rug=True, kde=False, bins=10)
sns.distplot(df_diff['next'], label='time to next', rug=True, kde=False, bins=10)
legend();


def plot_time_map(df_diff, coloring=None):
    cmap = plt.cm.get_cmap('RdYlBu')
    c = np.linspace(0, 1, len(df_diff)) if coloring is None else coloring
    scatter(df_diff['prev'], df_diff['next'],
            alpha=0.5,
            c=c,
            cmap=cmap,
            edgecolors='none')
    xlabel('time to previous event')
    ylabel('time to next event')
    title('Time map')
    axes().set_aspect('equal')
    
    max_value = df_diff[['prev','next']].max().max()
    plot([0, max_value], [0, max_value], alpha=0.1);
    xlim([0, max_value+0.1])
    ylim([0, max_value+0.1])

plot_time_map(df_diff);


def unique_chords(df):
    return sorted(df['chord'].unique())

for chord in unique_chords(df):
    print(chord)


import glob


files = glob.glob(data_dir + '*/*.lab')
tracks = pd.DataFrame({
    'album': [f.split('/')[-2].replace('_', ' ') for f in files],
    'name': [f.split('/')[-1].replace('.lab', '').replace('_', ' ') for f in files],
    'album_index': [int(f.split('/')[-2][:2]) for f in files]#,
#     'song_index': [int(f.split('/')[-1][:2]) for f in files]
})
tracks


def song_title(track):
    return ' / '.join(track[['album', 'name']])

def time_map_for_file(index):
    plot_time_map(add_differences(read_chord_file(files[index])).dropna())
    title(song_title(tracks.ix[index]))
    
time_map_for_file(5)


def add_track_id(df, track_id):
    df['track_id'] = track_id
    return df

selected_files = files
track_dfs = (read_chord_file(file) for file in selected_files)
track_dfs = (add_track_id(df, track_id) for (track_id, df) in enumerate(track_dfs))
track_dfs = (add_differences(df) for df in track_dfs)
all_events = pd.concat(track_dfs)
df_diff_all = all_events.dropna()


df_diff_all.head()


print('song count:', len(selected_files))
print('total diff event count in all songs:', len(df_diff_all))


df_diff_all.describe()


def outlier_quantiles(df, cols=['next','prev'], tolerance=0.01):
    df_nonzero = df[cols][df[cols] > 0]
    quantiles = df_nonzero.quantile([tolerance, 1 - tolerance])
    return quantiles

outlier_limits = outlier_quantiles(df_diff_all)
outlier_limits


def remove_outliers(df, limits, cols=['next','prev']):
    outlier_idxs = df['next'] == np.nan # just an array of False of proper length
    for col in cols:
        q_min, q_max = limits[col]
        print(q_min, q_max)
        series = df[col]
        idxs = series < q_min
        print(col, 'min', sum(idxs))
        outlier_idxs |= idxs
        idxs = series > q_max
        outlier_idxs |= idxs
        print(col, 'max', sum(idxs))
    print('outlier count:', sum(outlier_idxs), 'precentage:', sum(outlier_idxs) / len(df) * 100, '%')
    return df[~outlier_idxs]


df_diff_all_cleaned = remove_outliers(df_diff_all, outlier_limits)


df_diff_all_cleaned.describe()


plot_time_map(df_diff_all_cleaned, coloring=df_diff_all_cleaned['track_id'])


# It seems that velocity is represented by radius and acceleration by angle with range (0, pi/2).
# 
# Thus it might make sense to transform the time map via inverse polar transform so that velocity and acceleration are on cartesian coordinates.
# 

def inverse_polar(time_to_prev, time_to_next):
    # x = time_to_prev
    # y = time_to_next
    # (x, y) -> (r, phi) (cartesian to polar)
    # (r, phi) -> (velocity, acceleration) (no transform, just different interpretation)
    r = np.sqrt(time_to_prev**2 + time_to_next**2)
    phi = np.angle(time_to_next + 1j * time_to_prev) / (2 * np.pi)
    return (1 / (r / np.sqrt(2)), (phi - 0.125) * 8)

x = np.linspace(0, 1, 100)
plot(x, 1 - x)
scatter(*inverse_polar(x, 1 - x))
xlabel('velocity (r)')
ylabel('acceleration (phi)')
axes().set_aspect('equal');


def plot_inverse_polar_time_map(df_diff, coloring=None):
    cmap = plt.cm.get_cmap('RdYlBu')
    velocity, acceleration = inverse_polar(df_diff['prev'], df_diff['next'])
    c = np.linspace(0, 1, len(df_diff)) if coloring is None else coloring
    scatter(velocity, acceleration,
            alpha=0.5,
            c=c,
            cmap=cmap,
            edgecolors='none')
    xlabel('velocity')
    ylabel('acceleration')
    title('Time map')
    axes().set_aspect('equal')
    
    max_velocity = velocity.max()
    plot([0, 0], [max_velocity, 0], alpha=0.2);
    xlim([0, max_velocity+0.1])
    ylim([-1, 1])


plot_inverse_polar_time_map(df_diff);


plot_inverse_polar_time_map(df_diff_all_cleaned);


def plot_tracks(df, col, track_order=None):
    track_id = df['track_id']
    y = track_id
    if track_order is not None:
        mapping = track_order.argsort()
        y = y.apply(lambda x: mapping[x])
    plot(df[col], y, 'g.', label=col, alpha=0.1)
    xlabel(col)
    ylabel('track')

plot_tracks(df_diff_all, 'start')


def select_time_range(df, start, end, col='start'):
    series = df[col]
    return df[(series >= start) & (series <= end)]

plot_tracks(select_time_range(df_diff_all, 0, 100), 'start')


plot_tracks(df_diff_all_cleaned, 'next')


sns.distplot(df_diff_all_cleaned['next']);


next_medians = df_diff_all.groupby('track_id')['next'].median()
next_medians.describe()


tracks['next_median'] = next_medians
tracks_by_next_median = next_medians.argsort()
tracks.ix[tracks_by_next_median]


# Tracks ordered by median time difference between events.
# 

plot_tracks(df_diff_all, 'start', tracks_by_next_median)


scatter(tracks.ix[tracks_by_next_median]['album_index'], next_medians)
xlabel('album index')
ylabel('median of time-to-next within a song');


df = pd.DataFrame({
        'album': list(tracks['album_index'][df_diff_all_cleaned['track_id']]),
        'duration': list(df_diff_all_cleaned['next'])})
sns.violinplot(data=df, x='album', y='duration')
title('Distribution of chord segment durations (time-to-next) for each album');


# Songs ordered by total length.
# 

total_lengths = df_diff_all.groupby('track_id').last()['end']


# indexes of last songs in each album
last_song_indexes = list(tracks[tracks['album_index'].diff() != 0].index)


scatter(np.arange(len(total_lengths)), total_lengths, c=tracks['album_index'], cmap=plt.cm.get_cmap('RdYlBu'))
for i in last_song_indexes:
    axvline(i, alpha=0.1)
title('Total length of songs')
xlabel('track id')
ylabel('length (sec)');


scatter(tracks['album_index'], total_lengths, c=tracks['album_index'], cmap=plt.cm.get_cmap('RdYlBu'))
title('Total length of songs')
xlabel('album index')
ylabel('length (sec)');


plot(sorted(total_lengths));
title('Songs ordered by total length')
xlabel('track id (reordered)')
ylabel('length (sec)');


total_lengths.describe()


print('shortest song:', total_lengths.min(), 'sec,', song_titles[total_lengths.argmin()])
print('longest song:', total_lengths.max(), 'sec,', song_titles[total_lengths.argmax()])


sns.distplot(total_lengths, bins=20)
axvline(total_lengths.median(), alpha=0.2)
xlabel('total length(sec)');


album_lengths = tracks.join(total_lengths).groupby('album_index').sum()['end']
album_lengths


stem(album_lengths)
title('Total album lengths');


chords = df_diff_all['chord'].value_counts()
print('unique chord count:', len(chords))
print('top 20 chords:')
chords[:20]


plot(chords)
title('chord frequency');





get_ipython().magic('matplotlib inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
from music21.duration import Duration
from music21.instrument import Instrument
from music21.note import Note, Rest
from music21.stream import Stream
from music21.tempo import MetronomeMark
import numpy as np
import os
import scipy.io.wavfile

from generate_audio_samples import make_instrument, write_midi
from midi2audio import FluidSynth


mpl.rc('figure', figsize=(20, 10))


midi_notes = np.arange(128)
instruments = np.arange(128)


def sweep_instrument(instrument_id, output_dir):
    s = Stream()
    duration = Duration(1.0)
    s.append(make_instrument(instrument_id))
    s.append(MetronomeMark(number=120))
    for midi_number in midi_notes:
        s.append(Note(midi=midi_number, duration=duration))
        s.append(Rest(duration=duration))
    os.makedirs(output_dir, exist_ok=True)
    midi_file, audio_file = [
        output_dir + '/instrument_{0:03d}.{1}'.format(instrument_id, ext)
        for ext in ['midi', 'wav']]
    write_midi(s, midi_file)
    print('instrument:', audio_file)
    FluidSynth().midi_to_audio(midi_file, audio_file)

def sweep_instruments(output_dir):
    for instrument_id in instruments:
        sweep_instrument(instrument_id, output_dir)


audio_dir = 'data/working/instrument-ranges'
sweep_instruments(audio_dir)


def analyze_instrument_rms(i, audio_dir):
    """
    Compute the RMS of each note in the synthesized signal for a single instrument.
    """
    fs, x = scipy.io.wavfile.read('{0}/instrument_{1:03d}.wav'.format(audio_dir, i))
    # convert from stereo to mono
    x = x.mean(axis=1)
    # cut the leading rest
    x = x[fs // 2:]
    # align the ending
    x = x[:len(x) // fs * fs]
    # split the notes
    x_notes = x.reshape(-1, fs)
    # RMS for each note
    x_notes_rms = np.sqrt((x_notes**2).mean(axis=1))
    return x_notes_rms


plt.plot(analyze_instrument_rms(1, audio_dir), '.-')
plt.title('power for each note')
plt.xlabel('MIDI tone')
plt.ylabel('RMS')
plt.xlim(0,127);


def analyze_rms_for_all_instruments(audio_dir):
    """
    Compute a matrix of RMS for each instrument and note.
    """
    return np.vstack([analyze_instrument_rms(i, audio_dir) for i in instruments])


x_rms_instruments_notes = analyze_rms_for_all_instruments(audio_dir)


plt.imshow(x_rms_instruments_notes, interpolation='none')
plt.suptitle('MIDI instruments range - RMS power')
plt.xlabel('MIDI note')
plt.ylabel('MIDI instrument')
plt.savefig('data/working/instrument_ranges_rms.png');


np.save('data/working/instrument_ranges_rms.npy', x_rms_instruments_notes)


# There's a peak at value around 1.0 which represents quiet.
# 

plt.hist(x_rms_instruments_notes[x_rms_instruments_notes <= 1].flatten(), 200);


plt.hist(x_rms_instruments_notes[x_rms_instruments_notes > 1].flatten(), 200);


# The range of instruments split into quiet (black) and sounding (white) regions. We can limit the pitches to the sounding ones.
# 

plt.imshow(x_rms_instruments_notes > 1, interpolation='none', cmap='gray')
plt.grid(True)
plt.suptitle('MIDI instruments range - RMS power')
plt.xlabel('MIDI note')
plt.ylabel('MIDI instrument')
plt.savefig('data/working/instrument_ranges_binary.png');


# # Split chord segments with keys to blocks
# 
# Instead of separate chords we want a context consisting a sequence of several chords. Then we use this chord blocks as an input and key as output for training a key classifier.
# 
# For simplicity we do a few things:
# - remove silent chords (`N` label) and silent keys
# - create fixed-size chord blocks such that each block spans a single key
# 

import sys

sys.path.append('../../tools/music-processing-experiments')

from analysis import split_block_times


split_block_times(200, 50, 10)





# # Plain sinusoid autoencoder
# 
# The goal is to encode a signal consisting of sinusoid samples and decode it back.
# 

get_ipython().magic('pylab inline')
import keras
import numpy as np


# Input signal. Single training example.
# 

t = np.arange(50).reshape(1, -1)
x = np.sin(2*np.pi/50*t)
print(x.shape)
plot(t[0], x[0]);


# Simple autoencoder of four layers: 50 -> 25 -> 12 -> 25 -> 50.
# 

from keras.models import Sequential
from keras.layers import containers
from keras.layers.core import Dense, AutoEncoder

encoder = containers.Sequential([Dense(25, input_dim=50), Dense(12)])
decoder = containers.Sequential([Dense(25, input_dim=12), Dense(50)])

model = Sequential()
model.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True))

model.compile(loss='mean_squared_error', optimizer='sgd')


# prediction with initial weight should be random
plot(model.predict(x)[0]);


# train the model and store the loss values as function of time
from loss_history import LossHistory
loss_history = LossHistory()
model.fit(x, x, nb_epoch=500, batch_size=1, callbacks=[loss_history])


plot(loss_history.losses);


plot(log10(loss_history.losses));


# The model fits the data quite nicely.
# 

plot(model.predict(x)[0])
plot(x[0]);


# The model is able to predict on noise-corrupted data.
# 

x_noised = x + 0.2 * np.random.random(len(x[0]))
plot(x_noised[0], label='input')
plot(model.predict(x_noised)[0], label='predicted')
legend();


# However the model does is not able to predict a sinusoid with different phase.
# 

x_shifted = np.cos(2*np.pi/50*t)
plot(x_shifted[0], label='input')
plot(model.predict(x_shifted)[0], label='predicted')
legend();


# The model is able to deal with scaled sinuoid, but the farther it is from the original amplitude, the more noise.
# 

x_scaled = 0.2 * x
plot(x_scaled[0], label='input')
plot(model.predict(x_scaled)[0], label='predicted')
legend();


# # Sinusoid autoencoder trained with multiple phases
# 
# Let's provide more training examples - sinusoid with various phases.
# 

get_ipython().magic('pylab inline')
import keras
import numpy as np
import keras


N = 50
# phase_step = 1 / (2 * np.pi)
t = np.arange(50)
phases = np.linspace(0, 1, N) * 2 * np.pi
x = np.array([np.sin(2 * np.pi / N * t + phi) for phi in phases])
print(x.shape)
imshow(x);


plot(x[0]);
plot(x[1]);
plot(x[2]);


from keras.models import Sequential
from keras.layers import containers
from keras.layers.core import Dense, AutoEncoder

encoder = containers.Sequential([
        Dense(25, input_dim=50),
        Dense(12)
    ])
decoder = containers.Sequential([
        Dense(25, input_dim=12),
        Dense(50)
    ])

model = Sequential()
model.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True))

model.compile(loss='mean_squared_error', optimizer='sgd')


plot(model.predict(x)[0]);


from loss_history import LossHistory
loss_history = LossHistory()
model.fit(x, x, nb_epoch=1000, batch_size=50, callbacks=[loss_history])


plot(model.predict(x)[0])
plot(x[0])


plot(model.predict(x)[10])
plot(x[10])


print('last loss:', loss_history.losses[-1])
plot(loss_history.losses);


imshow(model.get_weights()[0], interpolation='nearest', cmap='gray');


imshow(model.get_weights()[2], interpolation='nearest', cmap='gray');


# The model should be able to handle noise-corrupted input signal.
# 

x_noised = x + 0.2 * np.random.random(len(x[0]))
plot(x_noised[0], label='input')
plot(model.predict(x_noised)[0], label='predicted')
legend();


# This time the model should be able to handle also phase-shifted signal since it was trained such.
# 

x_shifted = np.cos(2*np.pi/N * t.reshape(1, -1))
plot(x_shifted[0], label='input')
plot(model.predict(x_shifted)[0], label='predicted')
legend();





