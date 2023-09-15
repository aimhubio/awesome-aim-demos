import os

os.environ["OPENAI_API_KEY"] = ""

from llama_index import StorageContext, load_index_from_storage

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="index")

# load index
index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()
res = query_engine.query(
    "who is pual graham? what he suggests about building companies?"
)

print(res)
