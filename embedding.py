from openai import OpenAI
import pandas as pd
from scipy import spatial


api_key = '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'

client = OpenAI(api_key=api_key)

MODEL="gpt-4o"


def create_toolbox_embeddings(

        client,

        df,

        embedding_model: str = "text-embedding-3-large",

) -> pd.DataFrame:
    """Generates embeddings from a toolbox CSV file and returns a DataFrame.

    Args:

        client (OpenAI): OpenAI API client for generating embeddings.

        csv_path (str): Path to the CSV file.

        embedding_model (str): The embedding model to use.

    Returns:

        pd.DataFrame: DataFrame containing the text and embeddings.

    """

    # df = _merge_csv_text(csv_path)

    docs = df['tag']

    embeddings = []

    response = client.embeddings.create(model=embedding_model, input=docs)

    for i, be in enumerate(response.data):
        assert i == be.index  # double check embeddings are in same order as input

    batch_embeddings = [e.embedding for e in response.data]

    embeddings.extend(batch_embeddings)

    df_out = pd.DataFrame({"text": docs, "embedding": embeddings})

    df_out["metadata"] = [{"document": "sample txt"}] * len(df)

    output_path = "toolbox-text-with-embeddings.csv"

    # logging.info(f"\nSaving embeddings to {output_path}")

    df_out.to_csv(output_path, index=False)

    return df_out

df = pd.DataFrame(index = range(2), columns=['tag'])

df['tag'][0] = 'Personalized information is transferred from a first hand portable phone'
df['tag'][1] = ('The data transfer application on said computer is controlled to read said personalized '
                'information from said first memory means to store the personalized information in said '
                'first hand portable phone')


df_temp = create_toolbox_embeddings(

    client,

    df,

    "text-embedding-3-large",

)

def strings_ranked_by_relatedness(
    client: OpenAI,
    query: str,
    embeddings_df: pd.DataFrame,
    embedding_model: str = "text-embedding-3-large",
    top_n: int = 10,
) -> pd.DataFrame:
    """Ranks strings in a DataFrame by their relatedness to a query.

    Args:
        client (OpenAI): OpenAI API client for generating embeddings.
        query (str): The query string to compare against.
        embeddings_df (pd.DataFrame): DataFrame containing embeddings.
        embedding_model (str): The embedding model to use.
        top_n (int): Number of top results to return.

    Returns:
        pd.DataFrame: DataFrame of top n rows ranked by relatedness.
    """
    out_df = embeddings_df.copy()

    query_embedding_response = client.embeddings.create(
        model=embedding_model,
        input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding

    out_df["relatedness"] = out_df.apply(
        lambda row: 1 - spatial.distance.cosine(query_embedding, row["embedding"]),
        axis=1,
    )
    out_df = out_df.sort_values("relatedness", ascending=False)[:top_n]

    return out_df

query = 'Personalized information is transferred'
out_df = strings_ranked_by_relatedness(client,query, df_temp, "text-embedding-3-large",10)
print('Done')