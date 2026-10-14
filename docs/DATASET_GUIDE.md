# Dataset setup and governance

HAM10000 official references: [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000),
[Tschandl et al., Scientific Data 2018](https://doi.org/10.1038/sdata.2018.161),
[Harvard Dataverse record](https://doi.org/10.7910/DVN/DBW86T).
The original paper was read from the supplied PDF; its article CC BY licence does not itself
establish the image licence. A supplied CC BY-NC-SA deed does not establish which local files
it covers. Verify the exact downloaded record/version image terms, attribution, noncommercial
and share-alike obligations before publishing or redistributing. No images are committed.
Local metadata extracted from the existing archive; no dataset download is performed automatically.

Expected layout: archive/HAM10000_metadata.csv and image part directories. Required columns:
image_id, lesion_id, dx. Official metadata also has dx_type, age, sex, localization; optional
patient_id is used if genuinely present. Seven class map: akiec,bcc,bkl,df,mel,nv,vasc.
Multiple images can share lesion IDs. Patient IDs are generally unavailable; lesion grouping
cannot guarantee patient independence. No trustworthy skin-tone annotations are supplied by HAM10000.
Validation blocks malformed rows, corrupt/missing images, ambiguous filenames and conflicting
labels for exact duplicates. Nothing is silently excluded. A manifest contains paths and hashes,
not image bytes; treat manifests and error records as potentially sensitive too.

Future [DDI](https://aimi.stanford.edu/datasets/ddi-diverse-dermatology-images): obtain data through
the official process, verify agreement and redistribution restrictions before use. No active adapter.
Future [SCIN](https://github.com/google-research-datasets/scin): its repository specifies a SCIN
Data Use License and distinct self-reported/estimated FST and estimated Monk annotations. Review
that licence and annotation quality before integration. No active adapter or automatic download.

New adapters must document real paths, diagnosis taxonomy, image modality, patient/lesion grouping,
tone method/provenance/missingness, source/version, licence and permitted use. Never assume matching
labels between datasets. Acquisition domain shift must be separately evaluated.
