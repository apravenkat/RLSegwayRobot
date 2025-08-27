import pandas as pd
import numpy as np
# Load your CSV normally (only 1 column now)
df = pd.read_csv("log.csv", header=None)
df = df.iloc[1:].reset_index(drop=True)

# Split the single column into 4 separate columns
df_split = df[0].str.split(",", expand=True)

# Assign column names
df_split.columns = ['s0', 's1', 'a0', 'terminal_flag']

# Convert numeric columns to float
df_split[['s0', 's1', 'a0']] = df_split[['s0', 's1', 'a0']].astype(float)

# Clean up terminal_flag
df_split['terminal_flag'] = df_split['terminal_flag'].str.strip().str.lower()

# Assuming df_split already exists and is cleaned
df = df_split.copy()

# Make sure terminal_flag is lowercase for consistency
df['terminal_flag'] = df['terminal_flag'].str.strip().str.lower()

keep_mask = []
in_terminated_segment = False
done_already_kept = False

for flag in df['terminal_flag']:
    if flag == 'done':
        if not done_already_kept:
            # Keep the first 'done' row
            keep_mask.append(True)
            done_already_kept = True
            in_terminated_segment = True
        else:
            # Remove subsequent 'done' rows until 'not done'
            keep_mask.append(False)
    elif flag == 'not done':
        # Keep this row and reset flags
        keep_mask.append(True)
        in_terminated_segment = False
        done_already_kept = False
    else:
        # For any other flag:
        # Remove rows while in terminated segment, else keep
        keep_mask.append(not in_terminated_segment)

df_cleaned = df[keep_mask].reset_index(drop=True)
df_cleaned.to_csv('CleanedLogTest.csv', index=False)