## Summary

<!-- What does this change do, and why? -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Refactor (no behaviour change)
- [ ] Documentation
- [ ] Build / CI

## How this was tested

<!-- Commands you ran and what you observed. For an ML change, say which
     dataset you validated against. Do not quote metrics you have not
     reproduced. -->

```
pytest
```

## Checklist

- [ ] `pytest` passes
- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] New behaviour is covered by tests; bug fixes have a regression test
- [ ] The preprocessing pipeline is still fitted on training data only
- [ ] No metrics or benchmark figures were added that I did not reproduce
- [ ] Documentation updated if user-facing behaviour changed
