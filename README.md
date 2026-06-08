# BirdWatch

This repository accompanies the paper [Time-frequency localization of bird calls in dense soundscapes](TODO_ARXIV_LINK) and contains two components:

| | |
|---|---|
| [`birdbox/`](birdbox/) | Julia toolkit to train and run YOLO-based bird call detectors on raw audio |
| [`birdwatch/`](birdwatch/) | Tool to annotate and analyze spectrograms in the browser |

---

## BirdBox

BirdBox localizes bird vocalizations in time and frequency from raw audio using custom-trained [YOLO11](https://docs.ultralytics.com/) models. It includes two detectors pre-trained on complex soundscapes from Singapore, along with tools to apply them to new recordings and train custom detectors.

See [`birdbox/README.md`](birdbox/README.md) for setup and usage.

## BirdWatch

BirdWatch is a browser-based annotation and analysis tool for spectrograms that supports bounding-box labeling, playback of selected time–frequency regions, and evaluation of model predictions, with all processing performed locally in the browser.

See [`birdwatch/README.md`](birdwatch/README.md) for usage, or open the app directly: **[BirdWatch →](https://simhex.github.io/birdwatch/)**

---

## Citation

```bibtex
@misc{hexeberg2026,
  title  = {Time-frequency localization of bird calls in dense soundscapes},
  author = {Hexeberg, Simen and Tong, Fanghui and Vishnu, Hari and Chitre, Mandar},
  year   = {2026},
  eprint = {TODO_ARXIV_ID},
  archivePrefix = {arXiv},
}
```

## License

MIT — see [`LICENSE`](LICENSE).
