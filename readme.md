# Ollama fine tuned Synthetic Data Generator


```
docker build --progress=plain -f Dockerfile -t basinpoc .
docker run --name basinpoc -e SEED=42 -e NUM_CONTRACTS=5 -e TOKEN_STANDARD=ERC20 -e OMP_NUM_THREADS=1 -e MALLOC_ARENA_MAX=2 basinpoc
```