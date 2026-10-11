"""Flat Pydantic model for optical-transceiver datasheet extraction."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TransceiverSpecs(BaseModel):
    """Extract only explicitly stated values; use None when data is absent."""

    # Unknown keys are extraction failures. Allowing them hides the common
    # failure mode where the model invents a convenient nested structure
    # instead of filling the contract used by downstream code.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    #Module identification — vendor and part number
    vendor: Optional[str] = Field(None, description="Transceiver manufacturer or brand.")
    model: Optional[list[str]] = Field(None, description="Product model or orderable part number.")
    form_factor: list[str] | None = Field(default=None, description="Form factor of the transceiver: QSFP28, QSFP-DD, QSFP+, OSFP, CFP2-DCO")
    standards_claimed: list[str] | None  = Field(
        default=None, description="MSA Standards claimed in the datasheet."
    )

    #Frequency and wavelength parameters
    wavelength_min_nm: Optional[List[float]] = Field(
        None, description="Minimum supported of optical wavelengths ranges in nm."
    )
    wavelength_max_nm: Optional[List[float]] = Field(
        None, description="Maximum supported optical wavelengths of ranges in nm."
    )
    wavelength_center_nm: Optional[List[float]] = Field(
        None, description="Nominal or center optical wavelengths of ranges in nm."
    )
    wavelength_band: list[str] | None = Field(
        default=None,
        description="spectrum bands:C-band, L-band, O-band, CWDM, LAN-WDM, grey, Super/Extended variants.",
    )
    frequency_min_thz: Optional[List[float]] = Field(
        None, description="Minimum supported optical frequency in THz."
    )
    frequency_max_thz: Optional[List[float]] = Field(
        None, description="Maximum supported optical frequency in THz."
    )
    channel_spacing_ghz: Optional[List[float]] = Field(
        default_factory=list, description="Supported or minimum channel spacing in GHz."
    )
    channel_total: List[int] | int | None = Field(
        default=None, description="Total number of channels supported."
    )


    #Physical parameters
    connector_type: list[str] | None = Field(
            default=None,
            description="Optical connector: LC, SC, MPO, FC, ST. It is not a form factor or MSA code.",
        )
    fiber_type:  list[str] | None = Field(
            default=None,
            description="Transmission media/fiber: SM/SMF, MMF, G.652/G.655.",
        )
    temp_min_c: float | None = Field(
            default=None,
            description="Min normal operating temperature in Celsius.",
        )
    temp_max_c: float | None = Field(
            default=None,
            description="Max normal operating temperature in Celsius.",
        )
    power_consumption_w: Optional[List[float]] = Field(
        default_factory=list,
            description="Electrical Power consumption in Watts.",
        )


    # interface details (e.g., 400GAUI-8)
    host_interface_name: list[str] | None = Field(
        default=None,
        description="Host MSA code names as per MSA standards or vendor datasheet.",
    )

    host_interface_id_hex: list[str] | None = Field(
            default=None,
            description="Host Electrical Interface IDs in hexadecimal. Not Media codes",
        )

    host_interface_id: list[int] | None = Field(
            default=None,
            description="Host Electrical Interface id in decimals",
        )

    media_interface_name: list[str] | None = Field(
        default=None,
        description="Media MSA code names as per MSA standards or vendor datasheet.",
    )

    media_interface_id_hex: list[str] | None = Field(
            default=None,
            description="Media Interface IDs in hexadecimal. Not Host codes",
        )

    media_interface_id: Optional[List[int]] = Field(
            default=None,
            description="Media Interface id in decimals.",
        )
    frame: list[str] | None = Field(
        default=None, description="Frames or combination and multiples of frames: i.e. 8x 100GAUI-1"
    )
    standards_code: list[str] | None = Field(
        default=None, description="application code or profile or format related to any MSA, specifications, standards or vendor: openzr, openroadm, or specific application code"
    )

    # Electrical and optical performance parameters
    baud_rate_gbaud: Optional[List[float]] = Field(
        None, description="symbol rates in GBaud."
    )
    bit_rate_gbps: Optional[List[float]] = Field(
        default_factory=list, description="data rate/speed/payload bit rate in Gbps or G. do not confuse with GHz or grid"
    )
    roll_off_percent: Optional[List[float]] = Field(
        None, description="Spectral roll-off factor in percent."
    )
    modulation_format: Optional[List[str]] = Field(
        default_factory=list, description="Supported modulation formats: NRZ, PAM, QPSK, 8QAM, 16QAM, 64QAM, etc."
    )
    reach_km: Optional[List[float]] = Field(
        default_factory=list, description="Maximum stated transmission reach in km."
    )
    tx_power_min_dbm: Optional[List[float]] = Field(
        None, description="Minimum transmitter output power in dBm."
    )
    tx_power_max_dbm: Optional[List[float]] = Field(
        None, description="Maximum transmitter output power in dBm. sometimes given with TX shaping"
    )
    tx_in_band_osnr_db: Optional[List[float]] = Field(
        None, description="Transmitter-generated in band OSNR in dB."
    )
    tx_out_of_band_osnr_db: Optional[List[float]] = Field(
        None, description="Transmitter-generated out of band OSNR in dB."
    )


    #Module-level RX capability envelope — worst-case hardware limits

    rx_sensitivity_dbm: Optional[List[float]] = Field(
        None, description="Minimum receiver power in dBm at the stated BER."
    )
    rx_osnr_db: Optional[List[float]] = Field(
        None, description="Required receiver OSNR threshold in dB."
    )

    rx_osnr_tolerance_db_max: Optional[List[float]] = Field(
        default=None,
        description="Max required OSNR for target BER in dB/0.1nm.",
    )

    rx_overload_dbm: Optional[List[float]] = Field(
        None, description="Maximum receiver input power in dBm before overload."
    )
    cd_tolerance_ps_nm: Optional[List[float]] = Field(
        None, description="Chromatic-dispersion tolerance or CDC in ps/nm."
    )
    pdl_db: Optional[List[float]] = Field(
        None, description="Polarization-dependent-loss tolerance in dB."
    )
    pmd_tolerance_ps: Optional[List[float]] = Field(
        None, description="PMD or DGD tolerance in ps."
    )



    # FEC configuration fields
    fec_type: Optional[List[str]] = Field(
        default_factory=list, description="Supported FEC types as stated: KP4, oFEC, RS-FEC, KP4, KP1, KP2, KP3, KP4, KP5, KP6, KP7, KP8, RS(528)"
    )
    fec_overhead_percent: Optional[List[float]] = Field(
        default_factory=list, description="FEC overhead in percent."
    )
    pre_fec_ber_threshold: Optional[List[float]] = Field(
         default_factory=list, description="Maximum correctable pre-FEC bit-error ratio."
    )
    post_fec_ber_target: Optional[List[float]] = Field(
        None, description="Target post-FEC bit-error ratio."
    )

    # ── Direct-detect only ────────────────────────────────────────────────────
    # These three are listed in MODE_FIELDS, so a datasheet may state one value
    # per mode; a scalar type would reject the whole record in that case.
    extinction_ratio_db: Optional[List[float]] = Field(
        default=None,
        description="Extinction ratio in dB.",
    )
    los_assert_dbm: Optional[List[float]] = Field(
        default=None,
        description="LOS (loss of signal) assert threshold in dBm.",
    )
    los_deassert_dbm: Optional[List[float]] = Field(
        default=None,
        description="LOS deassert threshold in dBm.",
    )



    notes: Optional[str] = Field(
        None, description="Conditions, qualifiers, ambiguities, values trade-off dependencies, or source wording about any parameter that can not be explained in the set schema."
    )
    provenance_datasheet_sections: str | list[str] | None = Field(
        None, description="Comma-separated list of section headings (from the Section Headings list above) that contained the extracted parameters."
    )
    suggestions: Optional[str] = Field(
        None, description="Any suggestions for improving the extraction or schema."
    )


TransceiverDatasheet = TransceiverSpecs


class ApplicationMode(BaseModel):
    """One selectable optical operating mode copied from one source-table row."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    label: str = Field(description="Short source-grounded label, e.g. OpenZR+ app 4.")
    application_code: str | None = Field(None, description="Application or CMIS code exactly as printed.")
    standard: str | None = Field(None, description="Applicable standard or MSA exactly as printed.")
    payload_gbps: float | None = Field(None, description="Payload or line rate in Gbps.")
    host_format: str | None = Field(None, description="Host format exactly as printed.")
    electrical_interface: str | None = Field(None, description="Electrical host interface exactly as printed.")
    fec_type: str | None = Field(None, description="FEC type exactly as printed.")
    modulation_format: str | None = Field(None, description="Modulation format exactly as printed.")
    reach_km: float | None = Field(None, description="Operating reach in km.")
    tx_power_min_dbm: float | None = Field(None, description="Minimum transmitter power in dBm.")
    tx_power_max_dbm: float | None = Field(None, description="Maximum transmitter power in dBm.")
    rx_sensitivity_dbm: float | None = Field(None, description="Receiver sensitivity in dBm.")
    rx_osnr_db: float | None = Field(None, description="Receiver OSNR value in dB.")
    rx_osnr_0_5db_penalty_db: float | None = Field(None, description="Receiver OSNR at 0.5 dB penalty.")
    rx_osnr_1db_penalty_db: float | None = Field(None, description="Receiver OSNR at 1 dB penalty.")
    cd_tolerance_ps_nm: float | None = Field(None, description="Chromatic-dispersion or CDC range in ps/nm.")
    notes: str | None = Field(None, description="Only source conditions or qualifiers for this row.")


class ApplicationModeSet(BaseModel):
    """Schema-constrained response for a datasheet's application-mode tables."""

    model_config = ConfigDict(extra="forbid")
    modes: list[ApplicationMode] = Field(default_factory=list)

__all__ = [
    "ApplicationMode",
    "ApplicationModeSet",
    "TransceiverDatasheet",
    "TransceiverSpecs",
]

FIELD_BUNDLES = {
    "identification": ["vendor", "model", "form_factor", "standards_claimed"],
    "interfaces": ["host_interface_name", "host_interface_id_hex", "media_interface_name", "media_interface_id_hex"],
    "wavelength": ["wavelength_min_nm", "wavelength_max_nm", "wavelength_center_nm", "wavelength_band", "channel_spacing_ghz"],
    "physical": ["connector_type", "fiber_type", "temp_min_c", "temp_max_c"],
    "electrical": ["baud_rate_gbaud", "bit_rate_gbps", "modulation_format"],
    "fec": ["fec_type", "fec_overhead_percent", "pre_fec_ber_threshold", "post_fec_ber_target"],
}

# Fields that participate in per-mode configuration (can have multiple correlated values)
MODE_FIELDS: list[str] = [
    "channel_spacing_ghz",
    "form_factor",
    "frequency_max_thz",
    "frequency_min_thz",
    "wavelength_band",
    "wavelength_center_nm",
    "wavelength_max_nm",
    "wavelength_min_nm",
    "channel_total",
    "model",
    "baud_rate_gbaud",
    "bit_rate_gbps",
    "roll_off_percent",
    "modulation_format",
    "reach_km",
    "tx_power_min_dbm",
    "tx_power_max_dbm",
    "tx_in_band_osnr_db",
    "tx_out_of_band_osnr_db",
    "rx_sensitivity_dbm",
    "rx_osnr_db",
    "rx_osnr_tolerance_db_max",
    "rx_overload_dbm",
    "cd_tolerance_ps_nm",
    "pdl_db",
    "pmd_tolerance_ps",
    "fec_type",
    "fec_overhead_percent",
    "pre_fec_ber_threshold",
    "post_fec_ber_target",
    "extinction_ratio_db",
    "los_assert_dbm",
    "los_deassert_dbm",
    "host_interface_name",
    "host_interface_id_hex",
    "host_interface_id",
    "media_interface_name",
    "media_interface_id_hex",
    "media_interface_id",
    "power_consumption_w",
    "standards_code",
    "frame",
]



# Fields that participate in per-mode configuration (can have multiple correlated values)
STANDARDS_IDENTITY_FIELDS: list[str] = [
    "host_interface_name",
    "host_interface_id_hex",
    "host_interface_id",
    "media_interface_name",
    "media_interface_id_hex",
    "media_interface_id",
    "standards_code",
    "frame",
]


# Fields that participate in per-mode configuration (can have multiple correlated values)
STANDARDS_IMPLIED_FIELDS: list[str] = [
    "channel_spacing_ghz",
    "baud_rate_gbaud",
    "bit_rate_gbps",
    "modulation_format",
    "reach_km"
]
