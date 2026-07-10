from enum import Enum
from dataclasses import dataclass
import textwrap
import shutil


class MetadataEnum(str, Enum):
    """Base enum type for metadata fields that have controlled vocabularies."""

@dataclass(frozen=True, slots=True)
class MetadataField:
    """Schema definition for one metadata field.

    Attributes:
        name: Canonical metadata key used in file storage.
        python_type: Expected Python type for primitive validation.
        required: Whether the field is mandatory for the metadata section.
        units_required: Whether a units suffix/value is required for this field.
        enum_type: Optional enum class used for controlled vocabularies.
        description: Optional human-readable field description.
    """

    name: str
    python_type: type
    required: bool
    units_required: bool = False
    enum_type: type[MetadataEnum] | None = None
    description: str = ""


class Type(Enum):
    """Top-level metadata sections defined by the brim schema."""

    Experiment = 'Experiment'
    Optics = 'Optics'
    Brillouin = 'Brillouin'
    Acquisition = 'Acquisition'
    Spectrometer = 'Spectrometer'


class ImmersionMedium(MetadataEnum):
    """Allowed values for `Optics.Immersion_medium`."""

    other = 'other'
    air = 'air'
    water = 'water'
    oil = 'oil'


class SignalType(MetadataEnum):
    """Allowed values for `Brillouin.Signal_type`."""

    other = 'other'
    spontaneous = 'spontaneous'
    stimulated = 'stimulated'
    time_resolved = 'time_resolved'


class PhononsMeasured(MetadataEnum):
    """Allowed values for `Brillouin.Phonons_measured`."""

    other = 'other'
    longitudinal = 'longitudinal-like'
    transverse = 'transverse-like'
    longitudinal_Transverse = 'longitudinal-transverse-like'


class PolarizationProbedAnalyzed(MetadataEnum):
    """Allowed values for `Brillouin.Polarization_probed_analyzed`."""

    other = 'other'
    VH = 'VH'
    HV = 'HV'
    HH = 'HH'
    VV = 'VV'
    V_Unpolarized = 'V-unpolarized'
    Circular_Circular = 'circular-circular'


class ScanningStrategy(MetadataEnum):
    """Allowed values for `Acquisition.Scanning_strategy`."""

    other = 'other'
    point_scanning = 'point_scanning'
    line_scanning = 'line_scanning'
    light_sheet = 'light_sheet'
    time_resolved = 'time_resolved'


class SpectrometerType(MetadataEnum):
    """Allowed values for `Spectrometer.Type`."""

    other = 'other'
    VIPA = 'VIPA'
    FP = 'Fabry_Perot'
    stimulated = 'stimulated'
    heterodyne = 'heterodyne'
    time_domain = 'time_domain'
    impulsive = 'impulsive'


class DetectorType(MetadataEnum):
    """Allowed values for `Spectrometer.Detector_type`."""

    other = 'other'
    EMCCD = 'EMCCD'
    CCD = 'CCD'
    sCMOS = 'sCMOS'
    PMT = 'PMT'
    balanced = 'balanced'
    single_PD = 'single_PD'
    single_APD = 'single_APD'


METADATA_SCHEMA: dict[Type, tuple[MetadataField, ...]] = {
    Type.Experiment: (
        MetadataField('Datetime', str, required=False, description="[ISO 8601](https://www.iso.org/iso-8601-date-and-time-format.html) datetime when the experiment was started"),
        MetadataField('Timestamp', float, required=False, description="When used as array metadata, it indicates the elapsed time in milliseconds from the start of the experiment (defined by the 'Datetime' attribute of the current 'Data_{n}' group, if available; otherwise an arbitrary reference) at which the current spectrum was acquired. When used as an attribute, it indicates the elapsed milliseconds from the general metadata 'Datetime'"),
        MetadataField('Temperature', float, required=False, units_required=True, description="the temperature measured as close as possible to the sample"),
        MetadataField('Temperature_uncertainty', float, required=False, units_required=True, description=""),
        MetadataField('Sample', str, required=False, description="description of the sample being imaged"),
        MetadataField('Info', str, required=False, description="any additional description that the user can input to describe the experiment"),
    ),
    Type.Optics: (
        MetadataField('Wavelength', float, required=True, units_required=True, description="wavelength of the laser used for the measurements"),
        MetadataField('Power', float, required=True, units_required=True, description="total optical power on the sample"),
        MetadataField('Resolution_x', float, required=True, units_required=True, description=""),
        MetadataField('Resolution_y', float, required=True, units_required=True, description=""),
        MetadataField('Resolution_z', float, required=True, units_required=True, description=""),
        MetadataField('Lens_NA', float, required=True, description="the numerical aperture of the lens that is used for imaging (detection)"),
        MetadataField('Lens_NA_illum', float, required=False, description="the numerical aperture of the lens that is used for illumination (if different from detection)"),
        MetadataField('Immersion_medium', str, required=True, enum_type=ImmersionMedium, description="the immersion medium used for the objective lens"),
        MetadataField('Objective_model', str, required=False, description="the description of the objective lens being used, including the manufacturer and magnification"),
        MetadataField('Laser_model', str, required=False, description=""),
    ),
    Type.Brillouin: (
        MetadataField('Signal_type', str, required=True, enum_type=SignalType, description=""),
        MetadataField('Scattering_angle', float, required=True, units_required=True, description="the average scattering angle (i.e. between the optical axes of the illumination and detection); 180deg corresponds to backscattering"),
        MetadataField('Phonons_measured', str, required=True, enum_type=PhononsMeasured, description=""),
        MetadataField(
            'Polarization_probed_analyzed',
            str,
            required=True,
            enum_type=PolarizationProbedAnalyzed,
            description=""
        ),
        MetadataField('Shift_precision', float, required=True, units_required=True, description=""),
        MetadataField('Width_precision', float, required=True, units_required=True, description=""),
    ),
    Type.Acquisition: (
        MetadataField('Scanning_strategy', str, required=True, enum_type=ScanningStrategy, description=""),
        MetadataField('Acquisition_time', float, required=True, units_required=True, description="the time that takes to acquire a single ‘unit’, which is different depending on the scanning strategy (i.e. point, line, plane, A-line, etc.)"),
    ),
    Type.Spectrometer: (
        MetadataField('IRF', list[float], required=False, description="a 1D array containing the impulse response function of the spectrometer. It must have an attribute ‘Frequency’ [float] (with the corresponding ‘Frequency_units’ [string]) of the same length, containing the frequency axis."),
        MetadataField('IRF_frequency', list[float], required=False, description="a 1D array containing the frequency axis for the IRF. It is required if the IRF is provided and must be of the same length as the IRF array. An attribute ‘IRF_frequency_units’ must also be defined."),
        MetadataField('Type', str, required=True, enum_type=SpectrometerType, description=""),
        MetadataField('Resolution', float, required=True, units_required=True, description=""),
        MetadataField('Detector_type', str, required=False, enum_type=DetectorType, description=""),
        MetadataField('Detector_model', str, required=False, description=""),
        MetadataField('Additional_filter', str, required=False, description="description of any additional filter present in the spectrometer (e.g. vapor cell, Lyot stop, etc..)"),
        MetadataField('Confocal_pinhole_diameter', float, required=False, units_required=True, description=""),
    ),
}

def schema_as_string(
    include_description: bool = True,
    description_width: int | None = None,
    attr_width: int = 30,
    type_width: int | None = None,
    mandatory_width: int = 10    
) -> str:
    """Return metadata schema formatting as a string.

    Output is grouped by metadata section and includes field type,
    enum values (if any), and whether each field is mandatory.

    Args:
        include_description: Whether to include the description column.
        description_width: Column width for descriptions. If ``None``, computed
            from terminal width with a sensible minimum when descriptions are shown.
        attr_width: Column width for attribute names.
        type_width: Column width for type information. If ``None``, computed
            from terminal width with a sensible minimum.
        mandatory_width: Column width for mandatory flag.
    """
    sections = METADATA_SCHEMA
    if not sections:
        return "No metadata schema defined."

    terminal_columns = shutil.get_terminal_size(fallback=(120, 20)).columns

    min_attr_width = 12
    min_mandatory_width = 9
    min_type_width = 16
    min_description_width = 16

    attr_width = max(min_attr_width, attr_width)
    mandatory_width = max(min_mandatory_width, mandatory_width)

    if include_description:
        separators = 3
        fixed_width = attr_width + mandatory_width + separators
        available_for_dynamic = max(32, terminal_columns - fixed_width)

        if type_width is None and description_width is None:
            type_width = max(min_type_width, int(available_for_dynamic * 0.55))
            description_width = max(min_description_width, available_for_dynamic - type_width)
        elif type_width is None:
            description_width = max(min_description_width, description_width)
            type_width = max(min_type_width, available_for_dynamic - description_width)
        elif description_width is None:
            type_width = max(min_type_width, type_width)
            description_width = max(min_description_width, available_for_dynamic - type_width)
        else:
            type_width = max(min_type_width, type_width)
            description_width = max(min_description_width, description_width)

        total_width = fixed_width + type_width + description_width
        if total_width > terminal_columns:
            overflow = total_width - terminal_columns
            reducible_description = max(0, description_width - min_description_width)
            reduce_from_description = min(overflow, reducible_description)
            description_width -= reduce_from_description
            overflow -= reduce_from_description

            if overflow > 0:
                reducible_type = max(0, type_width - min_type_width)
                reduce_from_type = min(overflow, reducible_type)
                type_width -= reduce_from_type
                overflow -= reduce_from_type
    else:
        separators = 2
        fixed_width = attr_width + mandatory_width + separators
        available_for_type = max(min_type_width, terminal_columns - fixed_width)
        if type_width is None:
            type_width = available_for_type
        else:
            type_width = max(min_type_width, min(type_width, available_for_type))

    lines: list[str] = []

    for section_type, fields in sections.items():
        lines.append("")
        lines.append(section_type.value)
        lines.append("-" * len(section_type.value))
        if include_description:
            lines.append(
                f"{'Attribute':<{attr_width}} {'Type':<{type_width}} {'Mandatory':<{mandatory_width}} {'Description':<{description_width}}"
            )
            lines.append(
                f"{'-' * attr_width} {'-' * type_width} {'-' * mandatory_width} {'-' * description_width}"
            )
        else:
            lines.append(f"{'Attribute':<{attr_width}} {'Type':<{type_width}} {'Mandatory':<{mandatory_width}}")
            lines.append(f"{'-' * attr_width} {'-' * type_width} {'-' * mandatory_width}")

        for field in fields:
            type_label = field.python_type.__name__
            if field.enum_type is not None:
                enum_values = ", ".join([repr(item.value) for item in field.enum_type])
                type_label = f"enum {field.enum_type.__name__} ({enum_values})"

            mandatory_label = "yes" if field.required else "no"
            wrapped_type = textwrap.wrap(type_label, width=type_width) or [type_label]
            if include_description:
                description_label = field.description or ""
                description_width = int(description_width)
                wrapped_description = textwrap.wrap(description_label, width=description_width) or [""]
                line_count = max(len(wrapped_type), len(wrapped_description))
                for i in range(line_count):
                    attr_text = field.name if i == 0 else ""
                    mandatory_text = mandatory_label if i == 0 else ""
                    type_line = wrapped_type[i] if i < len(wrapped_type) else ""
                    description_line = wrapped_description[i] if i < len(wrapped_description) else ""
                    lines.append(
                        f"{attr_text:<{attr_width}} {type_line:<{type_width}} {mandatory_text:<{mandatory_width}} {description_line:<{description_width}}"
                    )
            else:
                for i, type_line in enumerate(wrapped_type):
                    attr_text = field.name if i == 0 else ""
                    mandatory_text = mandatory_label if i == 0 else ""
                    lines.append(
                        f"{attr_text:<{attr_width}} {type_line:<{type_width}} {mandatory_text:<{mandatory_width}}"
                    )

    return "\n".join(lines).lstrip("\n")


def print_schema(
    include_description: bool = True, *, 
    description_width: int | None = None,
    attr_width: int = 30,
    type_width: int | None = None,
    mandatory_width: int = 10
) -> None:
    """Print all metadata attributes defined in ``METADATA_SCHEMA``.
    Args:
        include_description: Whether to include the description column.
        For other arguments, see `schema_as_string()`. They are passed to this function to allow configuring the output format when printing the metadata schema.
    """
    
    print(
        schema_as_string(
            include_description=include_description,
            description_width=description_width,
            attr_width=attr_width,
            type_width=type_width,
            mandatory_width=mandatory_width
        )
    )