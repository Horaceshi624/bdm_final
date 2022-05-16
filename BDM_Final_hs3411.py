import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import IPython
from pyspark.sql.functions import col, split
from pyproj import Transformer
%matplotlib inline
IPython.display.set_matplotlib_formats('svg')
pd.plotting.register_matplotlib_converters()
sns.set_style("whitegrid")

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
sc = pyspark.SparkContext.getOrCreate()
spark = SparkSession(sc)

if __name__ == "__main__":
    ns=spark.read.load('nyc_supermarkets.csv',format ='csv',header = True,inferSchema = False)
    wpn=spark.read.load('/tmp/bdm/weekly-patterns-nyc-2019-2020',format ='csv',header = True,inferSchema = False)
    ns=ns.select(ns['safegraph_placekey'].alias('placekey'))

    wpn_1=ns.join(wpn,['placekey'],how = 'inner')
    month_list=['03','10']
    year_list=['2019','2020']

    wpn_2=wpn_1.select('placekey',F.split(wpn_1.date_range_start,"-").getItem(1).alias('month_start'),
                       F.split(wpn_1.date_range_start,"-").getItem(0).alias('year_start'),
                       F.split(wpn_1.date_range_end,"-").getItem(1).alias('month_end'),
                       F.split(wpn_1.date_range_end,"-").getItem(0).alias('year_end'),
                       'poi_cbg',
                       F.split(wpn_1.visitor_home_cbgs,'"').getItem(3).alias('visitor_home_cbgs'),
                       F.split(F.split(wpn_1.visitor_home_cbgs,':').getItem(1),'}').getItem(0).alias('visitor_num')                   
                       )
    wpn_2=wpn_2.filter(wpn_2['month_start'].isin(month_list))
    wpn_2=wpn_2.filter(wpn_2['month_end'].isin(month_list))
    wpn_2=wpn_2.filter(wpn_2['year_start']==wpn_2['year_end'])
    wpn_2=wpn_2.filter(wpn_2['year_start'].isin(year_list))

    nyc_list=['36061','36005','36047','36081','36085']
    wpn_3=wpn_2.filter(wpn_2['poi_cbg'][0:5].isin(nyc_list))

    ncc=spark.read.load('nyc_cbg_centroids.csv',format ='csv',header = True,inferSchema = False)
    ncc=ncc.select(ncc['cbg_fips'].alias('poi_cbg'),'latitude','longitude')
    wpn_3=wpn_3.join(ncc,['poi_cbg'],how = 'inner')
    ncc2=ncc.select(ncc['poi_cbg'].alias('visitor_home_cbgs'),ncc['latitude'].alias('latitude2'),ncc['longitude'].alias('longitude2'))
    wpn_3=wpn_3.join(ncc2,['visitor_home_cbgs'],how = 'inner')

    pd_df=wpn_3.toPandas()
    pd_df['distance']=((t.transform(pd_df['latitude'],pd_df['longitude'])[0]-t.transform(pd_df['latitude2'],pd_df['longitude2'])[0])**2 + (t.transform(pd_df['latitude'],pd_df['longitude'])[1]-t.transform(pd_df['latitude2'],pd_df['longitude2'])[1])**2)**0.5/5280
    pd_df['time']=pd_df['year_start']+'-'+pd_df['month_start']
    value = pd_df.values.tolist()
    column = list(pd_df.columns)
    wpn_4 = spark.createDataFrame(value,column)
    wpn_4=wpn_4.withColumn('totaldistance',col('distance')*col('visitor_num'))
    wpn_4=wpn_4.select(wpn_4['poi_cbg'].alias('cbg_fips'),'time',wpn_4['visitor_num'].cast('int'),'totaldistance')
    wpn_4=wpn_4.groupBy('cbg_fips','time').sum('totaldistance','visitor_num').sort('cbg_fips')
    wpn_4=wpn_4.withColumn('avgdistance',col('sum(totaldistance)')/col('sum(visitor_num)'))
    wpn_5=wpn_4.groupBy('cbg_fips').pivot('time').agg(F.first('avgdistance'))

    output = wpn_5.fillna('').sort('cbg_fips')
    output.write.options(header='true').csv(sys.argv[1])

