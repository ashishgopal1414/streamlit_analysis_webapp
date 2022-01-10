# Core Packages
import io, os, shutil
import numpy as np
from os import path
import missingno as msno

import streamlit as st

# EDA Packages
import pandas as pd
import numpy as numpy

# Data Viz Packages
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import missingno as msno
from pandas_profiling import ProfileReport
import io, os, shutil, re, time, pickle, pathlib, glob, base64, xlsxwriter
from io import BytesIO

## Disable Warning
st.set_option('deprecation.showfileUploaderEncoding', False)
# st.set_option('deprecation.showPyplotGlobalUse', False)
#%%

data_flag = 0

#%%
current_path = os.getcwd()

## Create sub directories if not created: "Raw Data" , "Batch Wise Data" , "Aggregated Data"
folder_names = [name for name in ["Raw Data" , "Modified Data"]]

for folder_name in folder_names:
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
#%%

datafile_path = os.path.join(current_path, "Raw Data", "data.csv")
modifiedfile_path = os.path.join(current_path, "Modified Data", "data.csv")

data_df = pd.DataFrame()

################################################################################
# from typing import Dict
# @st.cache(allow_output_mutation=True)
# def get_static_store() -> Dict:
#     """This dictionary is initialized once and can be used to store the files uploaded"""
#     return {}

################################################################################

##@st.cache(suppress_st_warning=True)
def load_data():
    data_df = pd.DataFrame()
    if path.exists(datafile_path):
        data_df = pd.read_csv(datafile_path)
        if st.checkbox("Click to view data"):
            st.write(data_df)
    return data_df

################################################################################

def load_modified_data():
    data_df = pd.DataFrame()
    if (not os.path.exists(datafile_path)) & (os.path.exists(modifiedfile_path)):
        os.remove(modifiedfile_path)
    if path.exists(modifiedfile_path):
        data_df = pd.read_csv(modifiedfile_path)
        if st.checkbox("Click to view Modified data"):
            st.write(data_df)
    return data_df
################################################################################

def preprocess_data(data_df):
    st.write('---------------------------------------------------')
    if not data_df.empty:
        all_columns = data_df.columns.to_list()
        if st.checkbox("Data Preprocess (Keep checked in to add steps)"):
            ## Receive a function to be called for Preprocessing
            df = data_df.copy()
            txt = st.text_area(
                "Provide lines of code in the given format to preprocess the data, otherwise leave it as commented",
                "## Consider the dataframe to be stored in 'df' variable\n" + \
                "## for e.g.\n" + \
                "## df['col_1'] = df['col_1'].astype('str')")
            if st.button("Finally, Click here to update the file"):
                exec(txt)
                if os.path.exists(modifiedfile_path):
                    os.remove(modifiedfile_path)
                df.to_csv(modifiedfile_path, index=False)
                st.success("New file created successfully under: {}".format(modifiedfile_path))
            if st.checkbox("Click to view Modified file"):
                if os.path.exists(modifiedfile_path):
                    st.write(pd.read_csv(modifiedfile_path))
                else:
                    st.markdown('**No Data Available to show!**.')

    else:
        st.markdown('**No Data Available to show!**.')
    st.write('---------------------------------------------------')

################################################################################
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index = True, sheet_name='Sheet1',float_format="%.2f")
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="Results.xlsx">Download Excel file</a>' # decode b'abc' => abc

def analysis_data(df):
    if not df.empty:
        df = df.copy()

        ## storing the data of Groups

        def data_grouping(data_df, feature_col, target_col):
            data_df = data_df.copy()
            cols_x = [feature_col]
            col_y = target_col

            for col_X in cols_x:
                print(
                    '##################################################################################################')
                print(f'{col_X} vs {col_y}')

                #     df_crosstab_overall = pd.DataFrame()
                df_crosstab_overall = pd.crosstab(index=data_df[col_X], columns=data_df[col_y])
                df_crosstab_overall.columns = ['EmpExit0', 'EmpExit1']
                df_crosstab_overall['Exit Ratio %'] = np.divide(df_crosstab_overall['EmpExit1'],
                                                                (df_crosstab_overall['EmpExit1'] + df_crosstab_overall[
                                                                    'EmpExit0'])) * 100

                df_crosstab_overall.sort_values(by='EmpExit1', ascending=False, inplace=True)

                overall_cols = ['Overall_EmpExit0', 'Overall_EmpExit1', 'Overall_ExitRatio%']

                ## Filtering Dataset
                for clus in list(data_df['Cluster'].unique()):
                    #         clus = 'Cluster 0' ## cluster 0
                    print(f'Cluster {clus}: ..............................................')
                    df_cluster = data_df[data_df['Cluster'] == clus].copy()
                    df_cluster.reset_index(drop=True, inplace=True)
                    # st.write(df_cluster)

                    ## Chi Sq. Test
                    ## if shape of crosstab >0 only then running the chi sq. test
                    df_crosstab = pd.crosstab(index=df_cluster[col_X], columns=df_cluster[col_y])
                    df_crosstab.columns = ['EmpExit0', 'EmpExit1']
                    df_crosstab['Exit Ratio %'] = np.divide(df_crosstab['EmpExit1'],
                                                            (df_crosstab['EmpExit1'] + df_crosstab['EmpExit0'])) * 100
                    df_crosstab.sort_values(by='EmpExit1', ascending=False, inplace=True)

                    # savepath = os.path.join(dirpath, clus)
                    fname_overall = f'{col_X} vs {col_y}.csv'
                    fname_overall = fname_overall.replace('/', '')
                    fname = fname_overall.replace('.csv', f'__{clus}.csv')
                    #         df_crosstab.reset_index(inplace=True)
                    overall_cols = overall_cols + [f'{clus}_' + col for col in df_crosstab.columns]
                    df_crosstab_overall = pd.concat([df_crosstab_overall, df_crosstab], ignore_index=True, axis=1)

                #         df_crosstab.to_csv(os.path.join(savepath,fname))

                df_crosstab_overall.columns = overall_cols
                df_crosstab_overall.sort_values(by='Overall_EmpExit1', ascending=False, inplace=True)
                #     df_crosstab_overall.to_csv(os.path.join(dirpath,fname_overall))
                return df_crosstab_overall

        ########################################################################################################
        def data_grouping2(data_df, feature_col, target_col):
            data_df = data_df.copy()
            cols_x = [feature_col]
            col_y = target_col
            # st.write(data_df)
            # st.write(data_df.shape)

            for col_X in cols_x:
                print('##################################################################################################')
                print(f'{col_X} vs {col_y}')

                # df_crosstab_overall = pd.DataFrame()
                df_crosstab_overall = pd.crosstab(index=data_df[col_X], columns=data_df[col_y])
                # st.write(df_crosstab_overall)
                # st.write(df_crosstab_overall.shape)
                # st.write(df_crosstab_overall.columns)

                for col in list(df_crosstab_overall.columns):
                    if len(df_crosstab_overall.columns) == 1:
                        if col == 1:
                            df_crosstab_overall['EmpExit0'] = np.nan
                            df_crosstab_overall['EmpExit1'] = df_crosstab_overall.iloc[:, 0]
                            df_crosstab_overall.drop(columns=col, inplace=True)
                        elif col == 0:
                            df_crosstab_overall['EmpExit0'] = df_crosstab_overall.iloc[:, 0]
                            df_crosstab_overall['EmpExit1'] = np.nan
                            df_crosstab_overall.drop(columns=col, inplace=True)
                    elif len(df_crosstab_overall.columns) == 2:
                        df_crosstab_overall.columns = ['EmpExit0', 'EmpExit1']

                # df_crosstab_overall.columns = ['EmpExit0', 'EmpExit1']
                df_crosstab_overall['Exit Ratio %'] = np.divide(df_crosstab_overall['EmpExit1'],
                                                                (df_crosstab_overall['EmpExit1'] + df_crosstab_overall[
                                                                    'EmpExit0'])) * 100

                df_crosstab_overall.sort_values(by='EmpExit1', ascending=False, inplace=True)

                overall_cols = ['FilteredOverall_EmpExit0', 'FilteredOverall_EmpExit1', 'FilteredOverall_ExitRatio%']
                # overall_cols = []

                ## Filtering Dataset
                for clus in list(data_df['Cluster'].unique()):
                    #         clus = 'Cluster 0' ## cluster 0
                    print(f'Cluster {clus}: ..............................................')
                    df_cluster = data_df[data_df['Cluster'] == clus].copy()
                    df_cluster.reset_index(drop=True, inplace=True)
                    # st.write(df_cluster)

                    ## Chi Sq. Test
                    ## if shape of crosstab >0 only then running the chi sq. test
                    df_crosstab = pd.crosstab(index=df_cluster[col_X], columns=df_cluster[col_y])

                    if df_crosstab.empty:
                        df_crosstab = pd.DataFrame(df_crosstab, columns=[0, 1])
                        df_crosstab.index.name = col_X
                    # st.write(df_crosstab)

                    for col in list(df_crosstab.columns):
                        # st.write(f'Length : {len(df_crosstab.columns)}')
                        # st.write(f'Columns: {col}')

                        if len(df_crosstab.columns)==1:
                            if col == 1:
                                df_crosstab['EmpExit0'] = np.nan
                                df_crosstab['EmpExit1'] = np.nan
                                df_crosstab['EmpExit1'] = df_crosstab.iloc[:,0]
                                df_crosstab.drop(columns=col, inplace=True)
                            elif col == 0:
                                df_crosstab['EmpExit0'] = np.nan
                                df_crosstab['EmpExit0'] = df_crosstab.iloc[:,0]
                                df_crosstab['EmpExit1'] = np.nan
                                df_crosstab.drop(columns=col, inplace=True)

                        elif len(df_crosstab.columns)==2:
                            df_crosstab.columns = ['EmpExit0', 'EmpExit1']

                    # st.write(df_crosstab)
                    # st.write(df_crosstab.columns)
                    # df_crosstab.columns = ['EmpExit0', 'EmpExit1']
                    try:
                        df_crosstab['Exit Ratio %'] = np.divide(df_crosstab['EmpExit1'],
                                                                (df_crosstab['EmpExit1'] + df_crosstab['EmpExit0'])) * 100
                    except Exception as e:
                        df_crosstab['Exit Ratio %'] = np.nan

                    df_crosstab.sort_values(by='EmpExit1', ascending=False, inplace=True)

                    # savepath = os.path.join(dirpath, clus)
                    fname_overall = f'{col_X} vs {col_y}.csv'
                    fname_overall = fname_overall.replace('/', '')
                    fname = fname_overall.replace('.csv', f'__{clus}.csv')
                    #         df_crosstab.reset_index(inplace=True)
                    overall_cols = overall_cols + [f'{clus}_' + col for col in df_crosstab.columns]
                    df_crosstab_overall = pd.concat([df_crosstab_overall, df_crosstab], ignore_index=True, axis=1)

                #         df_crosstab.to_csv(os.path.join(savepath,fname))

                df_crosstab_overall.columns = overall_cols
                # df_crosstab_overall.sort_values(by='Overall_EmpExit1', ascending=False, inplace=True)
                #     df_crosstab_overall.to_csv(os.path.join(dirpath,fname_overall))
                return df_crosstab_overall
        ########################################################################################################
        plotColor = ['b', 'g', 'r', 'm', 'c', 'y']
        markers = ['+', 'o', '*', '^', 'v', '>', '<']

        # set up
        sns.set(style='whitegrid')
        def Generate_bar_graph2(x, y, x_title, y_title, chart_title, color=plotColor):
            """ Based on x and y value, generate bar graph """

            fig, ax = plt.subplots(figsize=(15, 8))
            ax.bar(range(len(x))
                   , y
                   , color=color
                   , alpha=0.6)

            # stopping alphabetical sorting of graph
            plt.xticks(range(len(x)), x)
            plt.title(chart_title, fontsize=14)
            plt.xlabel(x_title, fontsize=13)
            plt.ylabel(y_title, fontsize=13)
            plt.grid(b=False)
            plt.yticks(fontsize=0)
            plt.xticks(rotation=45)
            plt.ylim(top=1)

            # # Visible x - axis line
            # for spine in plt.gca().spines.values():
            #     spine.set_visible(False) if spine.spine_type != 'bottom' else spine.set_visible(True)

            # Display label for each plot
            for i, v in (enumerate(y)):
                ax.text(i
                        , v + 0.05
                        , str(round((v * 100), 2)) + '%'
                        , fontsize=13
                        , ha='center')

            # plt.show()
            st.pyplot(fig)
        ########################################################################################################
        cluster_col = 'Cluster'
        target_col = 'Employee Exit'

        # select_target = st.selectbox('Select Target Column', df.columns)
        select_target = target_col

        Churn_rate = df[target_col].value_counts() / df.shape[0]

        Generate_bar_graph2(Churn_rate.index
                            , Churn_rate.values
                            , 'Employees'
                            , 'Percentage'
                            , 'Employee Distribution')
        # plt.show()
        st.write(Churn_rate)
        ##############################################################################################
        cluster_col      = 'Cluster'
        cluster_exit_col = 'Cluster__EmployeeExit'
        target_col       = 'Employee Exit'

        # select_target = st.selectbox('Select Target Column', df.columns)
        select_target = target_col

        Churn_rate = df[cluster_exit_col].value_counts() / df.shape[0]

        Generate_bar_graph2(Churn_rate.index
                            , Churn_rate.values
                            , 'Employees'
                            , 'Percentage'
                            , 'Employee Distribution')
        # plt.show()
        st.write(Churn_rate)
        ##############################################################################################

        selected_clusters = st.multiselect('Available Clusters', (list(df[cluster_col].unique())))
        if len(selected_clusters)>0:
            st.markdown(list(selected_clusters))
            df_filter = df[df[cluster_col].isin(selected_clusters)]
            st.write(df_filter.shape)
            st.write('##################################################################################################')
            select_dimension = st.selectbox('Select Dimension', (list(df_filter.columns)))

            if ((select_dimension is not None) & (select_target is not None)):
                st.write(f'{select_dimension} vs {select_target}')

                results_df = data_grouping(data_df=df,
                                           feature_col=select_dimension,
                                           target_col=select_target)
                st.write(results_df)
                st.markdown(str('<button type="button">' + get_table_download_link(results_df) + '</button>'),
                            unsafe_allow_html=True)
                ################################################################################
                select_dimension_val = st.multiselect('Select Dimension Value',
                                                    list(set(list(df_filter[select_dimension]))))

                if ((None not in select_dimension_val) and len(select_dimension_val)>0):
                    # st.write(f'{select_dimension} vs {select_target}')
                    df_filter2 = df_filter[df_filter[select_dimension].isin(select_dimension_val)]
                    st.write('##################################################################################################')

                    select_analysis = st.radio('Select Analysis Level', [None, 'Drilldown', 'Summarized'])
                    st.write(
                        '##################################################################################################')
                    if select_analysis == 'Summarized':
                        col1, col2, col3 = st.columns(3)
                        # col1, col2= st.columns(2)
                        select_dimension_summary = col1.multiselect('Select Dimensions for which analysis need to be summarized',
                                                                    (['All Dimensions ******'] + list(df_filter2.columns)))
                        # col1.write(select_dimension_summary)

                        top_k = int(col2.number_input('How many Top k rows required?', min_value=1, max_value=10, value=5, step=1))
                        # col2.write(top_k)

                        # min_thresh = col3.number_input('What is the minimum threshold for #of employees required?', 0, 1000, 20)
                        flag_avg = col3.checkbox('Select to filter only records > Average Value')

                        if ((st.checkbox('Click to Proceed Analysis')) & (len(select_dimension_summary) >0)):
                            for colx in select_dimension_val:
                                # st.write(f"Analysis for {select_dimension}=='{colx}' vs {select_target}")
                                df_filter3 = df_filter2[df_filter2[select_dimension] == colx].copy()

                                df_combined = pd.DataFrame()

                                if 'All Dimensions ******' in select_dimension_summary:
                                    select_dimension_summary = list(df_filter3.columns)


                                for col in select_dimension_summary:
                                    # st.write("--------------------------------------------------------------------")
                                    # st.write(f'{col} vs {select_target}')
                                    # st.write(df_filter3)
                                    results_df2 = data_grouping2(data_df=df_filter3,
                                                                 feature_col=col,
                                                                 target_col=select_target)
                                    results_df2.sort_values(by='FilteredOverall_EmpExit1', inplace=True, ascending=False)
                                    results_df2 = results_df2.reset_index()

                                    ## Flag to whether filter out records > avg for that column
                                    if flag_avg == True:
                                        # st.write(results_df2['FilteredOverall_EmpExit1'].mean())
                                        results_df2 = results_df2[results_df2['FilteredOverall_EmpExit1'] >= results_df2['FilteredOverall_EmpExit1'].mean()].copy()
                                        # st.write(results_df2)
                                    if results_df2.shape[0]>top_k:
                                        results_df2 = results_df2.head(top_k)
                                    # st.write(results_df2)
                                    # st.markdown(
                                    #     str('<button type="button">' + get_table_download_link(results_df2) + '</button>'),
                                    #     unsafe_allow_html=True)

                                    df_temp = results_df2.copy()
                                    df_temp['Dimension'] = col
                                    df_temp['Dimension_value'] = df_temp[col].astype(str)
                                    df_temp.drop(columns=col, inplace=True)
                                    cols = ['Dimension', 'Dimension_value'] + [cols for cols in list(df_temp.columns) if cols not in ['Dimension', 'Dimension_value']]

                                    if df_combined.empty:
                                        df_combined = df_temp.copy()
                                    else:
                                        df_combined = pd.concat([df_combined, df_temp], axis=0)

                                df_combined = df_combined[cols].copy()
                                st.write("--------------------------------------------------------------------")

                                st.markdown(f"<b>* * * Download Combined Results for {select_dimension}=='{colx}'! * * *</b>",unsafe_allow_html=True)

                                st.write(df_combined)
                                st.markdown(
                                    str('<button type="button">' + get_table_download_link(df_combined) + '</button>'),
                                    unsafe_allow_html=True)

                    elif select_analysis == 'Drilldown':
                        select_dimension2 = st.selectbox('Select Dimension2', df_filter2.columns)

                        if ((select_dimension2 is not None)):
                            st.write(f'{select_dimension2} vs {select_target}')
                            # st.write(df_filter2)
                            results_df2 = data_grouping2(data_df=df_filter2,
                                                         feature_col=select_dimension2,
                                                         target_col=select_target)
                            st.write(results_df2)
                            st.markdown(str('<button type="button">' + get_table_download_link(results_df2) + '</button>'),
                                        unsafe_allow_html=True)
                            ################################################################################
                            select_dimension_val2 = st.multiselect('Select Dimension2 Value',
                                                                 list(set(list(df_filter2[select_dimension2]))))

                            if ((None not in select_dimension_val2) and len(select_dimension_val2)>0):
                                # st.write(f'{select_dimension} vs {select_target}')
                                df_filter3 = df_filter2[df_filter2[select_dimension2].isin(select_dimension_val2)]
                                st.write(
                                    '##################################################################################################')
                                select_dimension3 = st.selectbox('Select Dimension3', df_filter3.columns)

                                if ((select_dimension3 is not None)):
                                    st.write(f'{select_dimension3} vs {select_target}')

                                    results_df3 = data_grouping2(data_df=df_filter3,
                                                                 feature_col=select_dimension3,
                                                                 target_col=select_target)
                                    st.write(results_df3)
                                    st.markdown(
                                        str('<button type="button">' + get_table_download_link(results_df3) + '</button>'),
                                        unsafe_allow_html=True)
                                    ################################################################################
                                    select_dimension_val3 = st.multiselect('Select Dimension3 Value',
                                                                         list(
                                                                             set(list(df_filter3[select_dimension3]))))

                                    if ((None not in select_dimension_val3) and len(select_dimension_val3)>0):
                                        # st.write(f'{select_dimension} vs {select_target}')
                                        df_filter4 = df_filter3[df_filter3[select_dimension3].isin(select_dimension_val3)]
                                        st.write(
                                            '##################################################################################################')
                                        select_dimension4 = st.selectbox('Select Dimension4', df_filter4.columns)

                                        if ((select_dimension4 is not None)):
                                            st.write(f'{select_dimension4} vs {select_target}')

                                            results_df4 = data_grouping2(data_df=df_filter4,
                                                                         feature_col=select_dimension4,
                                                                         target_col=select_target)
                                            st.write(results_df4)
                                            st.markdown(str('<button type="button">' + get_table_download_link(
                                                results_df4) + '</button>'),
                                                        unsafe_allow_html=True)
                                            ################################################################################
                                            select_dimension_val4 = st.multiselect('Select Dimension4 Value',
                                                                                 list(
                                                                                     set(list(df_filter4[select_dimension4]))))

                                            if ((None not in select_dimension_val4) and len(select_dimension_val4)>0):
                                                # st.write(f'{select_dimension} vs {select_target}')
                                                df_filter5 = df_filter4[df_filter4[select_dimension4].isin(select_dimension_val4)]
                                                st.write(
                                                    '##################################################################################################')
                                                select_dimension5 = st.selectbox('Select Dimension5', df_filter5.columns)

                                                if ((select_dimension5 is not None)):
                                                    st.write(f'{select_dimension5} vs {select_target}')

                                                    results_df5 = data_grouping2(data_df=df_filter5,
                                                                                 feature_col=select_dimension5,
                                                                                 target_col=select_target)
                                                    st.write(results_df5)
                                                    st.markdown(str('<button type="button">' + get_table_download_link(
                                                        results_df5) + '</button>'),
                                                                unsafe_allow_html=True)
                                                    ################################################################################
                                                    select_dimension_val5 = st.multiselect('Select Dimension5 Value',
                                                                                           list(
                                                                                               set(list(df_filter5[
                                                                                                            select_dimension5]))))

                                                    if ((None not in select_dimension_val5) and len(
                                                            select_dimension_val5) > 0):
                                                        # st.write(f'{select_dimension} vs {select_target}')
                                                        df_filter6 = df_filter5[
                                                            df_filter5[select_dimension5].isin(select_dimension_val5)]
                                                        st.write(
                                                            '##################################################################################################')
                                                        select_dimension6 = st.selectbox('Select Dimension6',
                                                                                         df_filter6.columns)

                                                        if ((select_dimension6 is not None)):
                                                            st.write(f'{select_dimension6} vs {select_target}')

                                                            results_df6 = data_grouping2(data_df=df_filter6,
                                                                                         feature_col=select_dimension6,
                                                                                         target_col=select_target)
                                                            st.write(results_df6)
                                                
                                                
                                                
    else:
        st.markdown('**No Data Available to show!**.')

    st.write('---------------------------------------------------')

################################################################################
# import SessionState
import time

def main():
    """ Semi Supervised Machine Learning App with Streamlit """
    # static_store = get_static_store()
    # session = SessionState.get(run_id=0)

    st.title("Data Science Webapp")
    #st.text("By Ashish Gopal")

    activities_outer = ["Data Ingestion", "Others", "About"]
    choice_1_outer = st.sidebar.radio("Choose your Step:", activities_outer)

    data = pd.DataFrame()

    if choice_1_outer == "Data Ingestion":
        file_types = ["csv","txt"]

        activities_1 = ["1. Data Import", "2. Data Preprocess", "3. Data Analysis"]
        choice_1 = st.sidebar.selectbox("Select Activities", activities_1)

        if choice_1 == "1. Data Import":
            data = None
            show_file = st.empty()
            if st.checkbox("Click to Upload data"):
                data = st.file_uploader("Upload Dataset : ",type=file_types)
            if not data:
                show_file.info("Please upload a file of type: " + ", ".join(file_types))
                if os.path.exists(datafile_path):
                    os.remove(datafile_path)
                return
            if data:
                if st.button("Click to delete data"):
                    if os.path.exists(datafile_path):
                        os.remove(datafile_path)
                        st.write('Raw File deleted successfully!')
                    elif os.path.exists(modifiedfile_path):
                        os.remove(modifiedfile_path)
                        st.write('Modified File deleted successfully!')
                    else:
                        st.write('No Files available for deletion!')
                    # static_store.clear()
                    data = None
                    # session.run_id += 1
                    return

                if data is not None:
                    df = pd.read_csv(data)
                    df.to_csv(datafile_path, index=False)
                    st.write('File loaded successfully!')
                if st.checkbox("Click to view data"):
                    if data is not None:
                        st.write(df)
                    else:
                        st.write('No Data available!')

        if choice_1 == "2. Data Preprocess":
            preprocess_data(load_data())

        if choice_1 == "3. Data Analysis":
            analysis_data(load_modified_data())


    if choice_1_outer == "Others":
        st.write('Coming Soon...')
        st.write('---------------------------------------------------')

    if choice_1_outer == "About":
        st.sidebar.header("About App")
        st.sidebar.info("Data Science Webapp")
        st.title("")
        st.title("")
        st.sidebar.header("About Developer")
        st.sidebar.info("https://www.linkedin.com/in/ashish-gopal-73824572/")
        st.subheader("About Me")
        st.text("Name: Ashish Gopal")
        st.text("Job Profile: Data Scientist")
        IMAGE_URL = "https://avatars0.githubusercontent.com/u/36658472?s=460&v=4"
        st.image(IMAGE_URL, use_column_width=True)
        st.markdown("LinkedIn: https://www.linkedin.com/in/ashish-gopal-73824572/")
        st.markdown("GitHub: https://github.com/ashishgopal1414")
        st.write('---------------------------------------------------')
if __name__ == '__main__':
	main()