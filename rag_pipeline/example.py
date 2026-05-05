import datetime as dt
from rag_pipeline.components.knowledgeLoader import WikipediaLoader
from rag_pipeline.pipeline import RAGPipeline
from rag_pipeline.components.chunker import BasicChunker
from rag_pipeline.components.embedders import SentenceTransformerEmbedder
from rag_pipeline.components.databases import FAISSDB
from rag_pipeline.components.retrievers import DenseRetriever
from rag_pipeline.components.rerankers import CrossEncoderReranker, PassthroughReranker
from rag_pipeline.components.generators import OllamaGenerator
from rag_pipeline.evaluate import EvalDataset, EvalSample, PipelineEvaluator, compare_reports
import pandas as pd
import os


job_id = os.environ.get("SLURM_JOB_ID") or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
# ──────────────────── Components for Pipeline ────────────────────
# Shared components
#data_path           = "data/wikiDump/psgs_w100.tsv"
data_path           = "data/wikiDump/psgs_sample_500.tsv"
chunker             = BasicChunker()
embedder            = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
vector_db = FAISSDB(
    dimension  = embedder.dimension,
    metric     = "cosine",
    use_gpu    = True,
    index_type = "ivf_pq",
    nprobe     = 64, # a non hardware performance knob, related to ivf_pq use vector_db.set_nprobe(128) to adjust it after training.
    m_pq       = 96, # related to ivf_pq.
    nbits_pq   = 8, # Standard is 8, lower gives smaller index and faster search but worse recall. Only relevant if m_pq > 0.
)

# Choose dataset to load, it chunks, embeds and indexes in batches.
knowledgeBaseLoader = WikipediaLoader(db=vector_db, embedder=embedder, chunker=chunker)

# Components for after database is built
retriever   = DenseRetriever(vector_db, retriever_top_k=30)
reranker    = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2", reranker_top_k=10)
generator   = OllamaGenerator("llama3.2")

# ──────────────────── Offline stage ────────────────────
# Offline stage. Only run once.
print("Loading and indexing data...")
vector_db = knowledgeBaseLoader.load_and_index(
    data_path,
    embed_batch_size=256,
    file_chunk_size=5_000,
    output_path=f"results/offline_{vector_db.__type__()}_{job_id}.json",
)

vector_db.save(f"results/{vector_db.__type__()}_{job_id}.index")

print("="*50)
print("Data loaded and indexed.")
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
                        reranker=PassthroughReranker(reranker_top_k=10),
                        generator=generator
                        )

# ──────────────────── Evaluate all three on the same dataset ────────────────────
#dataset = EvalDataset( # Natural Questions dataset
#    name="quick_eval",
#    samples=[
#        EvalSample(
#            query="What was the original name of the Academy Award for Best Production Design?",
#            gold_answer="Best Art Direction",
#        ),
#        EvalSample(
#            query="When did the Academy Award for Best Art Direction change its name?",
#            gold_answer="2012 for the 85th Academy Awards",
#        ),
#        EvalSample(
#            query="Since what year is the Academy Award for Best Production Design shared with set decorators?",
#            gold_answer="1947",
#        ),
#    ],
#)

df = pd.read_csv("data/NQ/Natural-Questions-Filtered-Subset.csv", sep=",")
#df = pd.read_csv("data/NQ/Natural-Questions-Filtered.csv", sep=",")
print(f"Number of samples in dataset: {len(df)}")
df = df.dropna(subset=["short_answers", "long_answers"], how="all") # Drop rows where both answers are missing
print(f"Number of samples after filtering: {len(df)}")

dataset = EvalDataset.from_dicts(
    name="natural_questions",
    items=[{
        "query":       row["question"],
        "gold_answer": row["short_answers"] if pd.notna(row["short_answers"]) 
                        else row["long_answers"],
        "metadata": {
            "all_answers": [
                a for a in [row.get("short_answers"), row.get("long_answers")]
                if pd.notna(a)
            ]}}
        for _, row in df.iterrows()
    ])

print(f"Dataset '{dataset.name}' loaded with {len(dataset.samples)} samples.")

pipeline_setup_1  = PipelineEvaluator(pipeline_1).run(dataset,  f"results/pipeline_1_{job_id}.json")
print("Pipeline 1 evaluation complete.")
pipeline_setup_2  = PipelineEvaluator(pipeline_2).run(dataset,  f"results/pipeline_2_{job_id}.json")
print("Pipeline 2 evaluation complete.")

print(compare_reports({
    "pipeline_1":  pipeline_setup_1,
    "pipeline_2":  pipeline_setup_2
}))