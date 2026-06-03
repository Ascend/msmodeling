# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import glob
import os
import re

import pandas as pd
from ..common import read_csv_s


def find_first_simulate_csv(input_path_2):
    # Check if the input path is a valid directory
    if not os.path.exists(input_path_2) or not os.path.isdir(input_path_2):
        raise NotADirectoryError("The provided path is not a valid directory.")

    # Build the search pattern
    pattern = os.path.join(input_path_2, "simulate*.csv")

    # Find all matching files
    files = glob.glob(pattern)

    # Check if there are matching files
    if not files:
        raise FileNotFoundError("No CSV files starting with 'simulate' found in the directory.")

    # Sort by file name
    files.sort()

    # Return the path of the first file
    return files[0]


def calculate_total_simulate_time(df_prefill, df3, df1):
    total_simulate_time = 0
    total_decode_simulate_time = 0

    for _, row in df_prefill.iterrows():
        digits = re.findall(r'\d+', row['reqinfo'])
        reqinfo_list = [int(num) for num in digits]
        non_zero_values = [x for i, x in enumerate(reqinfo_list) if i % 2 == 0]

        for val in non_zero_values:
            if pd.isna(df3.iloc[val]['reply_token_size']):
                continue
            during_time = df3.iloc[val]['first_token_latency']
            decode_time = df3.iloc[val]['execution_time(microsecond)'] - df3.iloc[val]['first_token_latency']
            arrive_time = row['start_time(microsecond)'] - during_time
            complete_time = row['start_time(microsecond)'] + decode_time

            filtered_df1 = df1[df1['start_time(microsecond)'] > arrive_time]
            filtered_df1 = filtered_df1[filtered_df1['start_time(microsecond)'] <= row['start_time(microsecond)']]

            filtered_df2 = df1[df1['start_time(microsecond)'] < complete_time]
            filtered_df2 = filtered_df2[filtered_df2['start_time(microsecond)'] >= row['start_time(microsecond)']]

            total_simulate_time += filtered_df1['simulate_time'].sum()
            total_decode_simulate_time += filtered_df2['simulate_time'].sum()

    return total_simulate_time, total_decode_simulate_time


def analyze(input_path_1, input_path_2):
    profiling_path = os.path.join(input_path_1, 'request.csv')
    df3 = read_csv_s(profiling_path, header=0)

    total_req = df3.shape[0]
    filtered_df = df3[df3['reply_token_size'].notna()]
    success_req = filtered_df.shape[0]

    batch_path = os.path.join(input_path_1, 'batch.csv')
    df1 = read_csv_s(batch_path, header=0)
    df1 = df1[df1['name'] == 'modelExec']
    simulate_path = find_first_simulate_csv(input_path_2)
    df2 = read_csv_s(simulate_path, header=None)
    column_name = ['simulate_time']
    df2.columns = column_name
    # Confirm that the two files have the same number of rows
    if len(df1) != len(df2):
        raise ValueError("The number of rows in the two CSV files must be the same")
    # Add the `during_time` column from the second CSV file to the first CSV file, and rename the column
    df1['simulate_time'] = df2['simulate_time'] / 10**6
    df3.sort_values(by='http_rid', ascending=True, inplace=True)
    df_prefill = df1[df1['batch_type'] == 'prefill']
    total_simulate_time, total_decode_simulate_time = calculate_total_simulate_time(df_prefill, df3, df1)

    df3['completed_time'] = df3['start_time_httpReq(microsecond)'] + df3['execution_time(microsecond)']
    total_latency = df3['first_token_latency'].sum() + total_simulate_time
    total_token = df3['reply_token_size'].sum()
    try:
        avg_prefill_latency = total_latency / success_req / 10**6
    except ZeroDivisionError as ex:
        raise ValueError(f"success_req cannot be zero. {ex}") from ex
    total_time = df3['completed_time'].max() - df3['start_time_httpReq(microsecond)'].min() + df1['simulate_time'].sum()
    try:
        throughput = total_token / total_time * 10**6
    except ZeroDivisionError as ex:
        raise ValueError(f"total_time cannot be zero. {ex}") from ex
    total_decode_time = (
        df3['execution_time(microsecond)'].sum() + total_decode_simulate_time - df3['first_token_latency'].sum()
    )
    try:
        average_decode_latency = total_decode_time / (total_token - success_req) / 10**6
    except ZeroDivisionError as ex:
        raise ValueError(f"total_token - success_req cannot be zero. {ex}") from ex
    try:
        success_precent = success_req / total_req
    except ZeroDivisionError as ex:
        raise ValueError(f"total_req cannot be zero. {ex}") from ex
    return throughput, avg_prefill_latency, average_decode_latency, success_precent
