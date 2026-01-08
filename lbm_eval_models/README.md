These are the guiding principles for how models are organized in lbm_eval. These
notes are largely intended for lbm_eval _developers_, but being able to
explicitly see the guiding principles may also be of use to lbm_eval users.
Be warned that some points may make reference to resources only available to
lbm_eval developers.

# Invariants for model organization

- All model species live in an exactly two-level hierarchy; e.g.,
  `groceries/apples` (anzu/lbm_eval/models/groceries/apples), à la Family-Genus.
  - There is never a third-level directory grouping of _models_. Only and always
    two levels.
- The first-level Family directory, `groceries`, never has any actual data; the
  most it would have is possibly a `README.md` file.
- The second-level Genus directory, `apples`, has multiple models (all kinds of
  apples, distribution shift apple models, etc.) but _only_ models.
  - For non-robot models, each model will be accompanied by a similarly named
    `README.md` file. The purpose of the `README.md` is to give a sense of what
    the model is even without pulling it up in a visualizer. A URL may be
    included. If it is a model of a _specific_ real-world object, a URL to the
    physical object will be provided. If the object is non-specific (e.g.,
    fruit), a URL to a reference image is given (typically the image referenced
    when creating the model). In some cases, no URL is possible.
- All of the model-supporting files (which should never be named directly in
  scenarios) are in an `assets` subdir of the Genus directory that uses them.
  That includes glTF files, textures, etc. and also the `foo_template.xacro`
  files from which model files are derived.
- The package URI for all of the moved models is `package://lbm_eval_models`
  (i.e., `package://lbm_eval_models/environment_maps/poly_haven_studio_4k.hdr`
  or `package://lbm_eval_models/groceries/apples/golden_delicious_apple.sdf`).
  Technically, these assets are likewise available via the Anzu package:
  `package://anzu/lbm_eval/models/groceries/apples/golden_delicious_apple.sdf`.
  This spelling should _never_ be used.
- Environment maps are special.
  - There is no Family-Genus hierarchy; there is just the `environment_maps`
    directory.
  - There is no `assets` subdirectory; the maps are stored directly in the
    `environment_maps` directory.

# Guidelines for introducing content

- There are specific guidelines for authoring new assets (glTF, textures, etc.)
  It can be found at:
  https://github.shared-services.aws.tri.global/robotics/anzu/blob/master/models/checklist_artist.md.
- New models must satisfy all legal requirements to be released as open source.
  TODO(sean.curtis): Make that true for all existing files. See
  https://docs.google.com/spreadsheets/d/1mVgFQqocf8TQiw1wbZzvB5pc9pXX7YVxo1B0AArusbk/edit?gid=0#gid=0

# License

The majority of LBM Eval is primarily distributed under the terms of both the
MIT license and the Apache License (Version 2.0); see the files LICENSE-MIT and
LICENSE-APACHE in this directory.

Small portions of LBM Eval are covered by other compatible open-source licenses
as indicated by the presence of a different LICENSE.TXT file in a subdirectory.

Any references or links to third-party products (including, but not limited to,
product pages) are provided solely for informational and convenience purposes.
We do not endorse, sponsor, or have any official affiliation with the
manufacturers, sellers, or platforms associated with these products. All product
names, logos, and brands are the property of their respective owners.
