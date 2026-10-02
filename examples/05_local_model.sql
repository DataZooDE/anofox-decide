-- examples/05_local_model.sql
-- Run a decision model fully inside DuckDB, offline, on your own exported ONNX graph (no key, no
-- network after setup). The graph and tokenizer are not part of the extension: export them once with
-- tools/export_julia (see "Local models" in the README), then point at the files.
-- offline: no (needs the exported files)

LOAD anofox_decide;

-- A model family is a "profile": 'julia-1' (default) or 'laya'. Laya also needs rl_agent_config.json
-- next to the graph.
SELECT decide_register_model('laya', 'local',
                             '/models/laya/julia1.onnx',
                             '/models/laya/tokenizer/tokenizer.json',
                             'laya');

-- Is it loadable? The doctor opens the files and tells you what is missing.
SELECT item, status, detail, fix FROM decide_doctor() WHERE detail LIKE '%laya%';

SELECT decide_probability('I want my money back.', 'A refund is requested.', model := 'laya') AS p;
