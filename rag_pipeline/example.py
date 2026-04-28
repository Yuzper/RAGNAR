from rag_pipeline.components.knowledgeLoader import WikipediaLoader
from rag_pipeline.pipeline import RAGPipeline
from rag_pipeline.components.chunker import BasicChunker
from rag_pipeline.components.embedders import SentenceTransformerEmbedder
from rag_pipeline.components.stores import FAISSDB
from rag_pipeline.components.retrievers import DenseRetriever
from rag_pipeline.components.rerankers import CrossEncoderReranker, PassthroughReranker
from rag_pipeline.components.generators import OllamaGenerator
from rag_pipeline.evaluate import EvalDataset, EvalSample, PipelineEvaluator, compare_reports

# ──────────────────── Components for Pipeline ────────────────────
# Shared components
data_path           = "psgs_sample_500.tsv"
chunker             = BasicChunker()
embedder            = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
vector_db           = FAISSDB(dimension=embedder.dimension, metric="cosine")
# Choose dataset to load, it chunks, embeds and indexes in batches.
knowledgeBaseLoader = WikipediaLoader(db=vector_db, embedder=embedder, chunker=chunker)

# Components for after database is built
retriever   = DenseRetriever(vector_db, retriever_top_k=20)
reranker    = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2", reranker_top_k=5)
generator   = OllamaGenerator("llama3.2")


# ──────────────────── Offline stage ────────────────────
# Offline stage. Only run once.
print("="*50)
print("Loading and indexing data...")
vector_db = knowledgeBaseLoader.load_and_index(data_path)

# ──────────────────── Online stage ────────────────────
# Pipelines to compare uses all same components exept what is swapped here.
pipeline_1 = RAGPipeline(chunker=chunker,
                        embedder=embedder,
                        DataBase=vector_db,
                        retriever=retriever,
                        reranker=reranker,
                        generator=generator,
                        )

pipeline_2 = RAGPipeline(chunker=chunker,
                        embedder=embedder,
                        DataBase=vector_db,
                        retriever=retriever,
                        reranker=reranker,
                        generator=generator
                        )

# ──────────────────── Evaluate all three on the same dataset ────────────────────
dataset = EvalDataset( # Natural Questions dataset
    name="quick_eval",
    samples=[
        EvalSample(
            query="What was the original name of the Academy Award for Best Production Design?",
            gold_answer="Best Art Direction",
        ),
        EvalSample(
            query="When did the Academy Award for Best Art Direction change its name?",
            gold_answer="2012 for the 85th Academy Awards",
        ),
        EvalSample(
            query="Since what year is the Academy Award for Best Production Design shared with set decorators?",
            gold_answer="1947",
        ),
    ],
)

pipeline_setup_1  = PipelineEvaluator(pipeline_1).run(dataset,  "results/pipeline_1.json")
pipeline_setup_2  = PipelineEvaluator(pipeline_2).run(dataset,  "results/pipeline_2.json")
pipeline_setup_3  = PipelineEvaluator(pipeline_2).run(dataset,  "results/pipeline_3.json")

print(compare_reports({
    "pipeline_1":  pipeline_setup_1,
    "pipeline_2":  pipeline_setup_2,
    "pipeline_3":  pipeline_setup_3
}))
