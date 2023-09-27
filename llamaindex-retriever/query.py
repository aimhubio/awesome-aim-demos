import os

os.environ["OPENAI_API_KEY"] = ""

from llama_index import StorageContext, load_index_from_storage, set_global_handler, ServiceContext
from llama_index.callbacks import CallbackManager

from aimstack.llamaindex_observer.callback_handlers import GenericCallbackHandler

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="index")

aim_cb = GenericCallbackHandler(repo='aim://0.0.0.0:8271')
callback_manager = CallbackManager([aim_cb])

service_context = ServiceContext.from_defaults(callback_manager=callback_manager)

# load index
index = load_index_from_storage(storage_context, service_context=service_context)

query_engine = index.as_query_engine()

res = query_engine.query(
    "How does Graham address the topic of competition and the importance (or lack thereof) of being the first mover in a market?"
)
aim_cb.flush()
print(res)
