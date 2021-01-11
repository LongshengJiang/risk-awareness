# Read required data into excel.
import re
import numpy as np
import seaborn
# Define the file name
filename = 'condition65_RTx.txt'
# Define the matching pattern using regular expression
re_pat_1 = re.compile(r'(Real found:)(\d+)(/)', re.I)
re_pat_2 = re.compile(r'(Human utilization: )(\d+)', re.I)
re_pat_3 = re.compile(r'(Time step: )(\d+)', re.I)
re_pat_4 = re.compile(r'(Average length of waiting line: )(\d+\.?\d+)', re.I)
# Define the variables
realfound = [None] * 20
human_uti = [None] * 20
search_time = [None] * 20
line_len = [None] * 20
# Define a loop counter
i = 0
with open(filename, 'r') as file:
    for line in file.readlines():
        match_pat_1 = re_pat_1.match(line)
        match_pat_2 = re_pat_2.match(line)
        match_pat_3 = re_pat_3.match(line)
        match_pat_4 = re_pat_4.match(line)
        if match_pat_1:
            # print(match_pat_1.group(2))
            realfound[i] = int(match_pat_1.group(2))
        if match_pat_2:
            # print(match_pat_2.group(2))
            human_uti[i] = int(match_pat_2.group(2))
        if match_pat_3:
            # print(match_pat_3.group(2))
            search_time[i] = int(match_pat_3.group(2))
        if match_pat_4:
            # print(match_pat_4.group(2))
            line_len[i] = float(match_pat_4.group(2))
            # print('loop:{}'.format(i))
            i += 1
print('realfound={}'.format(realfound))
print('human_utilization={}'.format(human_uti))
print('search_time={}'.format(search_time))
print('line_length={}'.format(line_len))
# Convert the list to numpy arrays
realfound = np.array(realfound)
human_uti = np.array(human_uti)
search_time = np.array(search_time)
line_len = np.array(line_len)
# Filter out the outliers based on search time. We use inter-quartile-range (IQR) to determine outliers
# (the concept used by boxplot).
q1 = np.percentile(search_time, 25)
q3 = np.percentile(search_time, 75)
iqr = q3 - q1
low_bound = q1 - 1.5 * iqr
up_bound = q3 + 1.5 * iqr
filter_bool = (low_bound <= search_time) & (search_time <= up_bound)
# Filtering the lists
realfound_flt = realfound[filter_bool]
human_uti_flt = human_uti[filter_bool]
search_time_flt = search_time[filter_bool]
line_len_flt = line_len[filter_bool]
# Print the number of outliers
num_outlier = len(search_time) - len(search_time_flt)
print('Number of outliers: {} \n'.format(num_outlier))
# Calculate the mean and standard deviation
print('filename: {} \n'.format(filename))
# real found
realfound_mean = realfound_flt.mean()
realfound_std = realfound_flt.std()
print('realfound_mean:{}'.format(realfound_mean))
print('realfound_std:{} \n'.format(realfound_std))
# human_utilization
human_uti_mean = human_uti_flt.mean()
human_uti_std = human_uti_flt.std()
print('human_uti_mean:{}'.format(human_uti_mean))
print('human_uti_std:{} \n'.format(human_uti_std))
# search time
search_time_mean = search_time_flt.mean()
search_time_std = search_time_flt.std()
print('search_time_mean:{}'.format(search_time_mean))
print('search_time_std:{} \n'.format(search_time_std))
# length of waiting line
line_len_mean = line_len_flt.mean()
line_len_std = line_len_flt.std()
print('line_len_mean:{}'.format(line_len_mean))
print('line_len_std:{} \n'.format(line_len_std))

#
print('end')
