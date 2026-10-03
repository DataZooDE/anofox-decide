-- examples/05_local_model.sql
-- Run a decision model fully inside DuckDB, offline, on your own machine: no key, and no network after the
-- one-time download of the model files from Hugging Face.
-- offline: no (the first line downloads about 0.7 GB)

LOAD anofox_decide;

-- Once: fetch and verify the weights of the multilingual Laya model into ~/.cache/anofox-decide
-- (SET anofox_decide_cache_dir = '<dir>' picks another place). Models: laya-multilingual,
-- laya-typed-decisions, julia-1.
CALL decide_download('laya-multilingual');

-- Which models can be called now, and what to run for the ones that cannot?
SELECT model, ready, hint FROM decide_models() WHERE model LIKE 'laya%' OR model = 'julia-1';

SELECT decide_probability('I want my money back.', 'A refund is requested.', model := 'laya-multilingual') AS p;

-- Your own exported ONNX graph (see "Your own exported model" in the README) is registered by hand:
-- SELECT decide_register_model('my-laya', 'local', '/models/laya/julia1.onnx',
--                              '/models/laya/tokenizer/tokenizer.json', 'laya');
