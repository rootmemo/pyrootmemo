# Notebook Execution Check

**Date:** 2026-06-26  
**Environment:** conda `test`

## Summary

| Status | Count |
|--------|-------|
| ✅ OK | 5 |
| ❌ FAILED | 13 |

---

## Failed Notebooks

### test_notebook.ipynb
**Error:** `TypeError: Parameter should be of type Parameter(value, unit)`

```
Cell In[3], line 1
----> 1 loam = Soil(name="clay_loam", unit_weight_bulk=1.20*9.81, friction_angle=37,cohesion=2.3)

File ~/git_folders/github/rootmemo/pyrootmemo/src/pyrootmemo/materials.py:375, in Soil.__init__
    374 if not is_namedtuple(v):
--> 375     raise TypeError("Parameter should be of type Parameter(value, unit)")

TypeError: Parameter should be of type Parameter(value, unit)
```

---

### test_notebook_fit_binned.ipynb
**Error:** `ModuleNotFoundError: No module named 'pyrootmemo.fit'`

```
Cell In[1], line 1
----> 1 from pyrootmemo.fit.fit_x_binned import PowerFitBinned

ModuleNotFoundError: No module named 'pyrootmemo.fit'
```

---

### test_notebook_powerlaw.ipynb
**Error:** `ModuleNotFoundError: No module named 'pyrootmemo.fit'`

```
Cell In[1], line 1
----> 1 from pyrootmemo.fit.fit_xy_powerlaw import PowerlawFitWeibull, ...

ModuleNotFoundError: No module named 'pyrootmemo.fit'
```

---

### test_notebook_profile.ipynb
**Error:** `TypeError: Groundwater table should be a single value`

```
Cell In[2], line 2
      1 silty_clay = Soil(name="silty clay")
----> 2 a = SoilProfile(soils=[silty_clay, silty_clay, silty_clay], depth=Parameter([1,1.3,3],"m"), groundwater_table=Parameter([1.5,2],"m"))

File ~/git_folders/github/rootmemo/pyrootmemo/src/pyrootmemo/geometry.py:88, in SoilProfile.__init__
     86 if k == "groundwater_table":
     87     if not isinstance(v.value, (float, int)):
---> 88         raise TypeError("Groundwater table should be a single value")

TypeError: Groundwater table should be a single value
```

---

### test_notebook_pullout.ipynb
**Error:** `ModuleNotFoundError: No module named 'pyrootmemo.pullout'`

```
Cell In[1], line 1
----> 1 from pyrootmemo.pullout import PulloutEmbeddedElastic, PulloutEmbeddedElasticSlipping

ModuleNotFoundError: No module named 'pyrootmemo.pullout'
```

---

### test_notebook_pullout_old.ipynb
**Error:** `ModuleNotFoundError: No module named 'pyrootmemo.pullout'`

```
Cell In[1], line 1
----> 1 from pyrootmemo.pullout import Pullout_embedded_elastic, Pullout_embedded_elastic_slipping, ...

ModuleNotFoundError: No module named 'pyrootmemo.pullout'
```

---

### test_notebook_rbmw.ipynb
**Error:** `AttributeError: 'Rbmw' object has no attribute 'peak_force'`

```
Cell In[4], line 8
----> 8 print(rbmw.peak_force())

AttributeError: 'Rbmw' object has no attribute 'peak_force'
```

---

### test_notebook_wwm.ipynb
**Error:** `ModuleNotFoundError: No module named 'pyrootmemo.wwm'`

```
Cell In[1], line 1
----> 1 from pyrootmemo.wwm import Wwm

ModuleNotFoundError: No module named 'pyrootmemo.wwm'
```

---

### fit_powerlaw_example.ipynb
**Error:** `ModuleNotFoundError: No module named 'pyrootmemo.fit'`

```
Cell In[2], line 1
----> 1 from pyrootmemo.fit import PowerlawFit

ModuleNotFoundError: No module named 'pyrootmemo.fit'
```

---

### test_notebook_fit_2D_normal_freesd_guess.ipynb
**Error:** `ModuleNotFoundError: No module named 'pyrootmemo.fit'`

```
Cell In[1], line 2
----> 2 from pyrootmemo.fit import LinearFit

ModuleNotFoundError: No module named 'pyrootmemo.fit'
```

---

### test_notebook_fit_new_1D.ipynb
**Error:** `ModuleNotFoundError: No module named 'pyrootmemo.fit'`

```
Cell In[1], line 2
----> 2 from pyrootmemo.fit import GammaFit, GumbelFit, WeibullFit, PowerFit

ModuleNotFoundError: No module named 'pyrootmemo.fit'
```

---

### test_notebook_fit_new_1D_class.ipynb
**Error:** `ModuleNotFoundError: No module named 'pyrootmemo.fit'`

```
Cell In[1], line 2
----> 2 from pyrootmemo.fit import calc_loglikelihood_power_class

ModuleNotFoundError: No module named 'pyrootmemo.fit'
```

---

### test_notebook_fit_new_2D.ipynb
**Error:** `ModuleNotFoundError: No module named 'pyrootmemo.fit'`

```
Cell In[1], line 2
----> 2 from pyrootmemo.fit import PowerlawFit, LinearFit

ModuleNotFoundError: No module named 'pyrootmemo.fit'
```

---

## Passed Notebooks

- `test_notebook_pullout_new.ipynb` ✅
- `test_notebook_stats_distribution.ipynb` ✅
- `test_notebook_stats_regression.ipynb` ✅
- `test_notebook_waldron.ipynb` ✅
- `test_notebook_fbm.ipynb` ✅
