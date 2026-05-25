# Plotting And Viewing

Static Matplotlib plot:

```python
mol.plot()
mol.plot(color_by="chain")
mol.plot(show_bonds=False)
```

Jupyter viewer with py3Dmol:

```python
viewer = mol.view(style="stick")
```

Animated spin GIF:

```python
mol.spin_gif("spin.gif")
```

Contact map heatmap:

```python
mol.plot_contact_map(cutoff=8.0, level="residue")
```
