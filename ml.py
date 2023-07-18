import os
from catboost import CatBoostClassifier
import pandas as pd
from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime
import numpy as np
from loguru import logger
from sqlalchemy import text
import hashlib
from fastapi import FastAPI
from typing import List
#from schema import PostGet
from pydantic import BaseModel


app = FastAPI()


class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True

class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]


def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH



def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
        logger.info(f"Got chunk: {len(chunk_dataframe)}")
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_features():

    logger.info('loading liked posts')
    liked_posts_query = text("""
        SELECT distinct post_id, user_id
        FROM public.feed_data
        where action = 'like'""")

    liked_posts = batch_load_sql(liked_posts_query)

    logger.info("loading posts features")
    # Фичи по постам на основе tf-idf
    posts_features = pd.read_sql_query(text(
        """SELECT * FROM public.velieva_posts"""),
        con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
            "postgres.lab.karpov.courses:6432/startml")


    logger.info("loading user features")
    user_features = pd.read_sql_query(text(
        """SELECT * FROM public.user_data"""),
        con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
            "postgres.lab.karpov.courses:6432/startml"
    )

    #print(liked_posts.shape, posts_features.shape, posts_features_dl.shape, user_features.shape)
    #return [liked_posts, posts_features_control, posts_features_test, user_features]
    return [liked_posts, posts_features, user_features]

def load_models():
    model_path = get_model_path("C:/Users/Admin/Start_ML/Deep learning/catboost_model_test")
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model

logger.info('loading model')
model_test = load_models()

logger.info('loading features')
features = load_features()
logger.info('service is up and running')



def get_recommended_feed(id: int, time: datetime, limit: int):

    #features[2].query('user_id == 1000').values[0] - cписок самих значений столбцов
    user_features = features[2].loc[features[2].user_id == id].drop('user_id', axis = 1)
    user_dict = dict(zip(user_features.columns, user_features.values[0]))

    post_features = features[1].drop(['index', 'text'], axis=1)
    for i in user_dict:
        post_features[i] = user_dict[i]

    post_features['hour'] = time.hour
    post_features['month'] = time.month

    post_features = post_features.reindex(columns=['user_id', 'post_id', 'topic', 'SUM_tfidf', 'MAX_tfidf', 'MEAN_tfidf',
                                               'claster_KMean', 'dist_1_claster', 'dist_2_claster', 'dist_3_claster',
                                               'dist_4_claster', 'dist_5_claster', 'dist_6_claster', 'dist_7_claster',
                                               'dist_8_claster', 'dist_9_claster', 'dist_10_claster', 'gender', 'age',
                                               'country', 'city', 'exp_group', 'os', 'source', 'hour', 'month'])

    content_test = features[1]
    # Выбираем нужную модель

    model = model_test

    predicts = model.predict_proba(post_features)[:, 1]
    liked_p_user = features[0][features[0].user_id == id].post_id.values
    # уберем те посты, которые были с лайком для данного юзера
    result = [i for i in post_features.post_id[np.argsort(predicts)].values if i not in liked_p_user][-limit:]


    result = [PostGet(**{'id': i,
            'text': content_test[content_test.post_id == i].text.values[0],
            'topic': content_test[content_test.post_id == i].topic.values[0]}) for i in result]


    return result

@app.get('/post/recommendations/', response_model = List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int) -> List[PostGet]:
    return get_recommended_feed(id, time, limit)

