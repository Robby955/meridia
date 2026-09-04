# Bar freeze report

- strong A on hidden-20260902: PASS
- strong B on hidden-20260902: PASS
- control register_only on hidden-20260902: FAILS accuracy, coverage, interval score, projection accuracy, projection coverage, projection interval score; e.g. accuracy: persons/nation worst error 0.0479 > 0.045047; interval score: persons/nation 0.7860 > 0.153284
- control survey_only on hidden-20260902: FAILS accuracy, coverage, interval score, projection accuracy, projection coverage, projection interval score; e.g. accuracy: persons/nation worst error 0.2464 > 0.045047; interval score: persons/nation 4.7919 > 0.153284
- control no_dedup on hidden-20260902: FAILS accuracy, coverage, interval score, projection accuracy, projection coverage, projection interval score; e.g. coverage: persons/all 0.390 < 0.7; coverage: households/all 0.268 < 0.7
- control inflated_intervals on hidden-20260902: FAILS interval score; e.g. interval score: persons/nation 0.7712 > 0.153284; interval score: persons/state 0.7704 > 0.652353
- control static_projection on hidden-20260902: FAILS projection accuracy, projection coverage, projection interval score; e.g. projection accuracy: children_under_16/nation worst error 0.1696 > 0.159909; projection interval score: children_under_16/nation 2.5786 > 0.472794
- control uniform_allocation on hidden-20260902: FAILS allocation; e.g. allocation: regret 0.2799 > 0.062915

RESULT: bars frozen; every control fails a named gate
