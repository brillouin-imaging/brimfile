from __future__ import annotations

from ..file_abstraction import FileAbstraction, sync, _gather_sync
from . import schema, validation
from .types import MetadataItem

from .. import units
from ..constants import brim_obj_names, reserved_attr_names

import warnings

import asyncio

__docformat__ = "google"


class Metadata:
    Item = MetadataItem
    Type = schema.Type

    def __init__(self, file: FileAbstraction, data_full_path: str = None):
        """
        Initialize the Metadata object.
        Args:
            file (FileAbstraction).
            data_full_path (str): The full path to the data group in the file. If None, only the metadata in the file are exposed.
        """
        self._file = file
        self._path = brim_obj_names.Brillouin_base_path
        self._general_metadata = None
        self._data_path = data_full_path

        #### version aware code ####

        ## _load_local_metadata ##        
        if self._file.version is None:
            # use latest version if the file version is not defined
            self._load_local_metadata = self._load_local_metadata_v0_2
        else:
            match self._file.version:
                case (0, 1, _):
                    self._load_local_metadata = self._load_local_metadata_v0_1
                case (0, 2, _):
                    self._load_local_metadata = self._load_local_metadata_v0_2

    @staticmethod
    def _raw_dict_to_MetadataItem_dict(raw_dict: dict) -> dict:
        """
        Convert a dictionary of raw metadata values (as read from the file)
        to a dictionary of MetadataItem objects.
        It also ignores the attributes that are defined in `reserved_attr_names`.
        Args:
            raw_dict (dict): A dictionary containing raw metadata values.
                It must be at depth 1, e.g. {"Temperature": 24, "Temperature_units": "C"}.
        Returns:
            dict: A dictionary containing MetadataItem objects, e.g. {"Temperature": MetadataItem(24, "C")}.
        """
        out_dict = {}
        for key in raw_dict.keys():
            if key not in reserved_attr_names:
                if not key.endswith('_units'):
                    u = raw_dict.get(f"{key}_units", None)
                    out_dict[key] = Metadata.Item(raw_dict[key], u)
        return out_dict

    async def _load_local_metadata(self, type: Type) -> dict:
        """
        Load the metadata of a specific type from the data group.
        This method is version-aware and must be overridden by version-specific implementation.
        Args:
            type (Type): The type of the metadata to load.
        Returns:    
        dict: A dictionary containing the metadata attributes of the specified type, where each element is of the type Metadata.Item.
        """
        # this is a placeholder method that will be overridden by version-specific implementations. It raises an exception if called directly.
        raise NotImplementedError(
            f"The add method is not implemented for this version {self._file.version} .brim file. Please use a version-specific implementation.")

    async def _load_local_metadata_v0_2(self, type: Type) -> dict:
        """
        See documentation of _load_local_metadata for details.
        """
        out_dict = {}
        if self._data_path is not None:
            metadata = await self._file.get_attr(self._data_path, 'Metadata')
            # read only the attributes of the specific type 
            metadata = metadata.get(type.value, {})
            out_dict = self._raw_dict_to_MetadataItem_dict(metadata)
        # TODO: implement reading _arrays
        return out_dict
    async def _load_local_metadata_v0_1(self, type: Type) -> dict:
        """
        See documentation of _load_local_metadata for details.
        """
        out_dict = {}
        if self._data_path is not None:
            attrs = await self._file.list_attributes(self._data_path)
            group = f"{type.value}."
            attrs = [attr for attr in attrs if attr.startswith(
                group) and not attr.endswith('_units')]
            coros_attrs = [self._file.get_attr(self._data_path, attr) for attr in attrs]
            coros_units = [units.of_attribute(self._file, self._data_path, attr) for attr in attrs]
            res = await asyncio.gather(*coros_attrs, *coros_units)
            for i, attr in enumerate(attrs):
                val = res[i]
                u = res[i + len(attrs)]
                out_dict[attr[len(group):]] = Metadata.Item(val, u)
        return out_dict
    
    async def _load_general_metadata(self):
        """
        Load the general metadata from the file's root attributes.
        If the general metadata has already been loaded, it returns a copy of the cached metadata.
        Returns:
            dict: A dictionary containing the general metadata attributes, as it is stored in the file.
            Note: The returned dictionary is a copy of the cached metadata to prevent accidental modifications to the data in the class.        
        """
        # TODO: implement handling of _arrays
        if self._general_metadata is not None:
            return self._general_metadata.copy()   
        metadata_dict = {}
        try:
            metadata_dict = await self._file.get_attr(self._path, 'Metadata')
        except Exception:
            # if the metadata group does not exist, create it
            for type in Metadata.Type:
                metadata_dict[type.value] = {}
            await self._file.create_attr(self._path, 'Metadata', metadata_dict)
        self._general_metadata = metadata_dict
        return metadata_dict.copy()
    

    async def _get_single_item(self, type: Type, name: str) -> Item:
        """
        Retrieve a single metadata.
        This method attempts to fetch a metadata attribute based on the specified type and name.
        If the instance is linked to a specific data group, it first checks if the metadata is
        defined within that group. If not, it retrieves the metadata from the general metadata group.
        Args:
            type (Type): The type of the metadata to retrieve.
            name (str): The name of the metadata attribute.
        Returns:
            The value of the requested metadata attribute and its units.
        Raises:
            KeyError: If the metadata attribute cannot be retrieved from either the specific
                       data group or the general metadata group.
        """

        metadata_dict = await self.to_dict_async(type)
        if name not in metadata_dict:
            raise KeyError(
                f"Metadata attribute {type.value}.{name} not found.")
        return metadata_dict.get(name)

    def __getitem__(self, key: str) -> Item:
        """
        Get the metadata for a specific key.
        Args:
            key (str): The key for the metadata. It has the format 'group.object', e.g. 'Experiment.Datetime'.
        """
        parts = key.split('.', 1)
        if len(parts) != 2:
            raise KeyError(
                f"Invalid key format: {key}. Expected 'group.object'.")
        group = parts[0]
        obj = parts[1]
        if group not in Metadata.Type.__members__:
            raise KeyError(
                f"Group {group} not valid. It must be one of {list(Metadata.Type.__members__)}")
        return sync(self._get_single_item(Metadata.Type[group], obj))

    def to_dict(self, type: Type, *, 
                validate: bool = False, include_missing: bool = False) -> dict:
        """
        Returns the metadata of a specific type as a dictionary. See doc of `to_dict_async`.
        """
        return sync(self.to_dict_async(type, validate=validate, include_missing=include_missing))
    async def to_dict_async(self, type: Type, *, 
                            validate: bool = False, include_missing: bool = False) -> dict | tuple[dict]:
        """
        Returns the metadata of a specific type as a dictionary.
        This method attempts to fetch a metadata attribute based on the specified type and name.
        If the instance is linked to a specific data group, it first checks if some metadata are
        defined within that group. If not, it retrieves the metadata from the general metadata group.
        Args:
            type (Type): The type of the metadata to retrieve.
            validate (bool): If True, validate the metadata against the schema.
            include_missing (bool): If True and validate is True, include missing metadata attributes in the result. Missing attributes will have the value None. If validate is False, this argument is ignored and missing attributes are not included in the result.
        Returns:
            dict: A dictionary containing all metadata attributes, where each element is of the type Item.
        """

        # load first the metadata defined locally in the data group (if the Metadata object is linked to a data group)
        local_metadata_dict = await self._load_local_metadata(type)
        if validate:
            # validate the local metadata first.
            for key, value in local_metadata_dict.items():
                _, local_metadata_dict[key] = validation.validate_single_field(type, key, value)

        # then load the general metadata from the metadata group
        global_metadata_dict = await self._load_general_metadata()
        global_metadata_dict = global_metadata_dict.get(type.value)
        global_metadata_dict = self._raw_dict_to_MetadataItem_dict(global_metadata_dict)
        if validate:
            # validate the general metadata.
            for key, value in global_metadata_dict.items():
                _, global_metadata_dict[key] = validation.validate_single_field(type, key, value)
        # make sure that we don't overwrite the global metadata dict in the class attribute.
        out_dict = global_metadata_dict.copy()
        # merge the local metadata dict into the global metadata dict. Local metadata takes precedence over global metadata.
        out_dict.update(local_metadata_dict)

        if validate and include_missing:
            # include missing required attributes with value None
            schema_attrs = [field.name for field in schema.METADATA_SCHEMA[type] if field.required]
            for attr in schema_attrs:
                if attr not in out_dict:
                    out_dict[attr] = Metadata.Item(None, None, validity=validation.MetadataItemValidity.MISSING_FIELD)

        return out_dict

    def add(self, type: Type, metadata: dict[str, Item], local: bool = False):
        """
        Add metadata to the file.
        Call `brimfile.metadata.schema.print_schema()` to see the list of available metadata attributes and their description.
        Args:
            type (Type): The type of the metadata to add.
            metadata (dict[str, Item]): A dictionary containing the metadata attributes to add.
                            Each element must be of type Metadata.Item.
                            The keys of the dictionary are the names of the attributes.
            local (bool): If True, the metadata will be added to the data group. Otherwise, it will be added to the general metadata group.
        """
        # to save local metadata, the Metadata object must be linked to a data group
        if local and self._data_path is None:
            raise ValueError(
                "The current metadata object is not linked to a data group. Set local to False to add the metadata to the general metadata group.")
        if not local:
            general_metadata = sync(self._load_general_metadata())        
        # iterate over the metadata dictionary and add each attribute
        for key, value in metadata.items():
            if not isinstance(value, Metadata.Item):
                # if no units are provided, we assume None
                value = Metadata.Item(value, None)
            key, value = validation.validate_single_field(type, key, value, report_on_invalid=True)
            val = value.value
            if local:
                group = self._data_path
                name = f"{type.value}.{key}"
                sync(self._file.create_attr(group, name, val))
                if value.units is not None:
                    units.add_to_attribute(self._file, group, name, value.units)
            else:
                general_metadata[type.value][key] = val
                if value.units is not None:
                    general_metadata[type.value][f"{key}_units"] = value.units
        if not local:
            self._general_metadata = general_metadata
            sync(self._file.create_attr(self._path, 'Metadata', general_metadata))

    def all_to_dict(self, *, 
                    validate: bool = False, include_missing: bool = False) -> dict:
        """
        Returns all the metadata as a dictionary.
        Returns:
            dict: A dictionary containing all the elements in Metadata.Type as a key.
                    Each of the key is defining a dictionary, as returned by Metadata.to_dict()
            For `validate and `include_missing` arguments, see `Metadata.to_dict_async()`.
        """
        types = [type for type in Metadata.Type]
        coros = [self.to_dict_async(type, validate=validate, include_missing=include_missing) for type in types]

        #retrieve all metadata asynchronously
        res = _gather_sync(*coros)
        #assign them to a dictionary
        full_metadata = {type.name: dic for type, dic in zip(types, res)}
        return full_metadata
    

# utility functions to retrieve specific metadata attributes, with unit conversion if necessary. 

    async def _get_wavelength_nm_async(self) -> float:
        """
            Retrieve the wavelength metadata and convert it to nm if necessary. If the wavelength metadata is not found, an exception is raised.
            If the wavelength metadata has no units defined, a warning is issued and the value is assumed to be in nm.
            Returns:
                float: The wavelength in nm.
            Raises:
                Exception: If the wavelength metadata cannot be retrieved or if the unit is not recognized.
        """
        wl = await self._get_single_item(Metadata.Type.Optics, 'Wavelength')
        if wl.units is None:
            warnings.warn(f"Wavelength metadata has no units defined. Assuming the value is in nm.")
            return float(wl.value)
        unit = str(wl.units).lower()
        value = float(wl.value)
        if unit in ('nm', 'nanometer', 'nanometers'):
            return value
        if unit in ('um', 'µm', 'micrometer', 'micrometers'):
            return value * 1e3
        if unit in ('m', 'meter', 'meters'):
            return value * 1e9
        raise ValueError(f"Unsupported wavelength unit '{wl.units}'.")

    async def _get_temperature_c_async(self) -> float:
        """
            Retrieve the temperature metadata and convert it to Celsius if necessary. If the temperature metadata is not found, an exception is raised.
            If the temperature metadata has no units defined, a warning is issued and the value is assumed to be in Celsius.
            Returns:
                float: The temperature in Celsius.
            Raises:
                Exception: If the temperature metadata cannot be retrieved or if the unit is not recognized.
        """    
        t = await self._get_single_item(Metadata.Type.Experiment, 'Temperature')       
        if t.units is None:
            warnings.warn(f"Temperature metadata has no units defined. Assuming the value is in Celsius.")
            return float(t.value)
        unit = str(t.units).lower()
        value = float(t.value)
        if unit in ('c', '°c', 'celsius'):
            return value
        if unit in ('k', 'kelvin'):
            return value - 273.15
        raise ValueError(f"Unsupported temperature unit '{t.units}'.")

    async def _get_scattering_angle_deg_async(self) -> float:
        """
            Retrieve the scattering angle metadata and convert it to degrees if necessary. If the scattering angle metadata is not found, an exception is raised.
            If the scattering angle metadata has no units defined, a warning is issued and the value is assumed to be in degrees.
            Returns:
                float: The scattering angle in degrees.
            Raises:
                Exception: If the scattering angle metadata cannot be retrieved or if the unit is not recognized.
        """
        sa = await self._get_single_item(Metadata.Type.Brillouin, 'Scattering_angle')
        if sa.units is None:
            warnings.warn(f"Scattering angle metadata has no units defined. Assuming the value is in degrees.")
            return float(sa.value)
        unit = str(sa.units).lower()
        value = float(sa.value)
        if unit in ('deg', '°', 'degree', 'degrees'):
            return value
        if unit in ('rad', 'radian', 'radians'):
            return value * 180 / 3.141592653589793
        raise ValueError(f"Unsupported scattering angle unit '{sa.units}'.")
      