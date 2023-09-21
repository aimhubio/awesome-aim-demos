import os

os.environ["OPENAI_API_KEY"] = ""

from llama_index import StorageContext, load_index_from_storage, set_global_handler, ServiceContext
from llama_index.callbacks import CallbackManager

from llamaindex_observer import AimGenericCallbackHandler

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="index")

aim_cb = AimGenericCallbackHandler(repo='aim://0.0.0.0:8271')
callback_manager = CallbackManager([aim_cb])

service_context = ServiceContext.from_defaults(callback_manager=callback_manager)

# load index
index = load_index_from_storage(storage_context, service_context=service_context)

query_engine = index.as_query_engine()

res = query_engine.query(
    "Who is Paul Graham? What he suggests about building companies?"
)

print(res)

print('\n'*10)

res = query_engine.query(
    "What Paul Graham thinks is the most reliable way to become a billionaire?"
)
print(res)
