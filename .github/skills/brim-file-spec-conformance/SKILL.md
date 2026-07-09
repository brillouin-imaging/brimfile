---
name: brim-file-spec-conformance
description: Use whenever writing, reviewing, or debugging brimfile test code or validation code that must confirm actual .brim file content (structure, attributes, array shapes, versioning across whichever brim_version(s) are currently implemented, sparse layout, metadata inheritance/overrides) agrees with the Brillouin-standard-file specification, rather than only with brimfile's own implementation. Also covers when to stop and ask instead of guessing — for ambiguous spec wording, or spec-correct expected behavior conflicting with actual behavior with no covering TODO.
license: MIT
---

# brim file spec conformance

The authoritative specification for the `.brim` file format lives in a **separate repository** from brimfile itself:
https://github.com/brillouin-imaging/Brillouin-standard-file. Do not rely on memory or on brimfile's implementation
alone to know what the format requires — fetch the current spec text when precision matters, because the spec has
changed across versions and brimfile does not necessarily implement every version's spec yet.

This skill is shared by (at least) two different kinds of work: writing **tests** that check brimfile's behavior
against the spec, and writing/extending the **validation code** (`src/brimfile/validation/`) that itself checks a
`.brim` file against the spec at runtime. The facts below apply equally to both; "Using this skill for validator
implementation work" at the end calls out the parts specific to writing validator code.

## Where to look

- The **un-prefixed files directly under `docs/`** — `brim_file_specs.md` (structure diagram + detailed
  field-by-field description of `/`, `/Brillouin_data`, `/Data_{n}`, `PSD`, `Frequency`, `Scanning`, `Parameters`,
  `Analysis_{m}`, `Calibration`), `brim_file_metadata.md` (metadata scopes, the hierarchical override/merge rule,
  and the full metadata schema — `Experiment`, `Optics`, `Brillouin`, `Acquisition`, `Spectrometer` — with
  required/optional and units-required flags), and `brim_file_subtypes.md` (the optional `Subtype`/`Subtype_features`
  mechanism, e.g. the `SinglePoint_VIPA` subtype implemented in `brimfile/subtypes/`) — describe the spec's
  **current** version. Read their own text to see which version number that is right now (as of this writing they
  open with "Proposal for version 0.2 of the brim file format" and give `'0.2'` as the example `brim_version`
  value). **Treat this as the final, authoritative spec for that version**, not as a tentative draft to wait on —
  the word "Proposal" in the heading reflects that it can still be edited (it isn't frozen into its own
  `docs/v{X.Y}/` folder yet), not that implementing/testing against it should wait until it is. Re-check which
  version this currently describes each time you use it — once it's eventually frozen into its own `docs/v0.2/`
  folder, the un-prefixed files will describe whatever version comes next.
- `docs/v{X.Y}/` — **frozen snapshots** of the same three files, kept for a version once a newer one has superseded
  it as "current" (e.g. `docs/v0.1/` exists because a later version is now the current one).
- `CHANGELOG.md` — the fastest way to see exactly what changed between versions (e.g. v0.1 → v0.2 changed measurement
  metadata overrides from flattened `Type.AttributeName` attributes to a nested `Metadata` JSON object, and added
  `_arrays` for per-position metadata). Always check this before writing a test that depends on version-specific
  behavior.
- `examples/v{X.Y}/` — example `.brim` files for a given spec version; useful as a ground-truth fixture when you need
  a realistic file rather than a synthetic minimal one.

## Discover which version(s) brimfile currently implements — don't assume there's only one

brimfile is meant to correctly support every `brim_version` it implements, and which versions that is **changes over
time** as new spec versions get implemented. Never hardcode an assumption like "only version X is supported" into a
test or into your understanding — that's exactly the kind of claim that goes stale as soon as support for another
version is added, and a stale claim like that will actively mislead you into under-testing (assuming a version isn't
supported when it now is) or asserting the wrong thing (assuming a behavior is version-independent when it's actually
version-specific).

Instead, derive the current picture fresh each time from the code, not from any cached summary (including this
skill file):
- Grep `src/brimfile/validation/main.py` (`validate_root_attrs` and any per-version branching) and
  `src/brimfile/file.py` (the `brim_version` default in `File.create`, and any other place `brim_version` is read
  or compared) to find the exact set of version string(s) currently recognized as valid.
- Check the spec repo for the full set of versions to support: every frozen `docs/v{X.Y}/` folder, **plus** whatever
  version the un-prefixed `docs/` files currently describe (read their text — see "Where to look" above). The
  current version counts as a version to support even before it's frozen into its own folder; don't treat it as
  optional or wait for that to happen.
- For each recognized version, check whether behavior that the spec's `CHANGELOG.md` says changed between versions
  (e.g. how metadata overrides are represented on disk) is actually implemented differently per version in the code,
  or whether the implementation currently only covers one representation. Don't assume either answer — check.
- Match each version's expected behavior against that exact version's spec text: a frozen `docs/v{X.Y}/` snapshot for
  a superseded version, or the un-prefixed `docs/` files for the current version. Both are equally authoritative for
  this purpose — the only difference is that the un-prefixed files can still change before being frozen, so re-check
  them if some time has passed since you last read them.

Write tests against — or implement version-strategy rules for — what's actually implemented/needed for each version,
parametrized (for tests) or registered (for validator code) over the discovered version list, so neither the tests
nor the validator need rewriting every time support for another version is added. If you find a genuinely
unimplemented gap (code claims to support a version but doesn't correctly implement part of that version's spec),
that's exactly the kind of thing to raise — see "Ambiguity and bugs" below.

## Checklist to apply when reviewing or writing a conformance test or validator check

Pull the exact wording from `brim_file_specs.md` / `brim_file_metadata.md` for whichever of these you're targeting,
rather than trusting this summary alone — this is a pointer list, not a substitute for reading the source doc:

- **Root attributes**: `brim_version` (required, semver string), `Subtype`/`Subtype_features` (optional, paired),
  `Authors`/`Lab` (optional).
- **`/Brillouin_data`**: must have a `Metadata` attribute (general metadata group).
- **`/Data_{n}`**: zero-based, contiguous-by-convention naming; must exist even for a single timepoint (`Data_0`);
  optional `Sparse` bool (default `False` if absent); `element_size` + `element_size_units` (required for non-sparse,
  optional for sparse); optional `Conditions`/`Conditions_name`.
- **`PSD`**: last dimension is spectral; for non-sparse, first three dims are `z, y, x`; for sparse, first dim is a
  flattened spatial index; extra dims (e.g. angle-resolved) go *between* spatial and spectral, never outside them.
- **`Frequency`**: must be broadcastable to `PSD`'s shape via NumPy broadcasting rules (trailing-dimension alignment).
- **`Scanning`**: `Spatial_map` (optional `x`/`y`/`z` arrays, units on the group not the arrays, missing axis = zeros)
  vs. `Cartesian_visualisation` (3D int index array, ZYX order, required attrs `element_size`/`element_size_units`,
  `-1` marks unused/non-cartesian pixels); `Cartesian_visualisation` is what makes sparse data reconstructible into a
  3D grid.
- **`Parameters`**: only relevant when `PSD` has extra dimensions; shape/naming rules differ by sparse vs. non-sparse
  (see spec — the dimension-count offset changes by exactly 1).
- **`Analysis_{m}`**: required `Shift_AS_{p}`/`Width_AS_{p}`/`Amplitude_AS_{p}`/`Offset_AS_{p}`; optional Stokes
  counterparts and `Fit_error_AS_{p}` subgroup; required `Fit_model` enum attribute
  (`other`/`Lorentzian`/`DHO`/`Voigt`); shapes must match `PSD`'s spatial dimensions.
- **`Calibration`**: optional; `Index` shape depends on sparse (1D) vs. non-sparse (3D), omittable if only one
  calibration spectrum; `Same_as` for de-duplication; nested `Metadata` (`Description`, `Temperature`, `Datetime`,
  `FSR`, `Timestamp`).
- **Metadata override/inheritance rule**: values defined lower in the hierarchy (e.g. `Data_{n}`) take precedence
  over values defined higher (e.g. `Brillouin_data`); the merge is meant to apply recursively at every level the spec
  defines a `Metadata` attribute for. Confirm exactly which levels brimfile currently implements this recursion for
  before asserting it works at a level you haven't checked (e.g. `Calibration`-level overrides may not be wired up
  even if `Data_{n}`-level ones are).
- **Naming/formatting rules that apply everywhere**: field names are case-sensitive and must match exactly; avoid
  symbol characters; datetimes are ISO 8601 strings; any attribute needing units gets a sibling `{name}_units`
  string attribute (or, for whole arrays, a `units` attribute on the array itself).

## Ambiguity and bugs: stop and ask, don't guess

Two situations that come up often when cross-checking against this spec require stopping and asking the user rather
than pushing through on your own judgment — whether the work at hand is a test or a change to the validator's own
implementation:

- **The spec wording is genuinely ambiguous** — e.g. it doesn't say what should happen in some combination of
  optional fields, it's unclear whether a rule stated for one group is meant to apply recursively to a nested one,
  or the current version's un-prefixed spec text contradicts itself in two passages (it can still be edited, so this
  happens more easily than in a frozen version). Don't silently choose a reading and encode it as a test's expected
  behavior or as a validator check's logic — state the ambiguity, cite the passage, and ask.
- **Spec-correct expected behavior conflicts with brimfile's actual behavior, and nothing in the code (no
  `TODO`/`FIXME` near the relevant path) flags this as known/expected.** That's a genuine implementation bug
  candidate, not something to quietly `xfail`, paper over, or silently "fix" as a side effect of unrelated work —
  stop and ask the user how they want to proceed before writing more tests or shipping more validator changes in
  that area. If a `TODO` *does* cover the gap, it isn't a surprise bug — note it for a deferred-coverage summary
  instead of stopping.

## Working method

1. Identify the exact spec clause(s) the work is meant to enforce; quote the section heading or a short paraphrase
   in the test docstring/comment, or in a code comment next to the check if you're writing validator code, so the
   spec ↔ code link survives future refactors.
2. Fetch the current text of the relevant spec file (don't rely on a cached mental summary) if there's any doubt —
   the spec is actively evolving (see `CHANGELOG.md`). `docs/v{X.Y}/` holds the frozen source of truth for a
   superseded version; the un-prefixed `docs/` files are the authoritative source for the current version even
   though they aren't frozen into their own folder yet.
3. Check brimfile's implementation to see whether it targets that spec version's behavior.
4. Write the test, or the validator check, against the implemented/intended behavior; note any gap versus the spec
   explicitly rather than papering over it.
5. Where the assertion concerns literal on-disk structure rather than an API return value, pair this skill with the
   `zarr-file-inspection` skill to verify at the zarr level too.

## Using this skill for validator implementation work (not just tests)

When the task is extending `src/brimfile/validation/` itself (rather than writing tests against it), apply the
sections above the same way, with one addition: before writing a check, decide whether it's **version-invariant**
(wording is the same across every version's spec text you checked — put it in the shared, version-agnostic call
graph) or **version-specific** (wording differs between versions — put only that delta in the affected version's
rules class, sourced from that version's own spec text: a frozen `docs/v{X.Y}/` snapshot for a superseded version,
or the un-prefixed `docs/` files for the current version, treated as final for this purpose even though they can
still be edited before being frozen). Don't default to assuming a check is version-invariant without having actually
compared the versions' wording — that assumption is exactly how a real per-version difference ends up silently
unimplemented for every version but the one whoever wrote the check happened to be looking at.
