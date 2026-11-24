"""VAT 时间相位、接缝、精度、实例和生命周期回归测试。"""

import math

import graphics.vat_vertex_anim as vat


def test_time_phase_is_independent_of_client_frame_rate():
    times = (0.0, 0.033333333333, 0.5, 1.0, 2.25)
    for t in times:
        reports = [
            vat.frame_coordinates(12, time=t, frame_rate=30.0)
            for _client_fps in (30, 60, 120, 144, 240)
        ]
        first = reports[0]
        for report in reports[1:]:
            assert report.phase == first.phase
            assert report.frame0 == first.frame0
            assert report.frame1 == first.frame1
            assert report.blend == first.blend

    mapped = vat.frame_rate_map(1.0, frame_count=12)
    assert mapped.identical_phase is True
    assert mapped.max_phase_delta == 0.0
    assert mapped.uses_client_frame_number is False


def test_determinism_report_is_stable():
    first = vat.determinism()
    second = vat.determinism()
    assert first.ok and second.ok
    assert first.identical_phase and second.identical_phase
    assert first.fingerprint == second.fingerprint


def test_loop_seam_wraps_last_frame_to_first():
    seam = vat.loop_seam(4, 3.7)
    assert seam.is_seam is True
    assert (seam.frame0, seam.frame1) == (3, 0)
    assert math.isclose(seam.blend, 0.7, abs_tol=1e-12)
    assert seam.wrap_mode == "first-last-wrap"


def test_loop_seam_pose_uses_wrapped_first_frame():
    texture = vat.VATTexture(frame_count=4, vertex_count=1)
    result = texture.sample(position=3.7, vertex=0)
    # 演示数据的 x 等于帧号；30% 第 3 帧 + 70% 第 0 帧 = 0.9。
    assert math.isclose(result.pose[0], 0.9, abs_tol=1e-9)


def test_non_loop_sample_clamps_to_end():
    result = vat.frame_coordinates(4, position=99.0, loop=False)
    assert (result.frame0, result.frame1, result.blend) == (3, 3, 0.0)


def test_normal_uses_separate_high_precision_storage():
    report = vat.normal_precision((0.2, -0.45, 0.87))
    assert report.separate_normal_storage is True
    assert report.block_compressed is False
    assert report.bits_per_channel == 16
    assert report.normal_format == "R16G16B16_SNORM"
    assert report.lighting_delta < 1e-3


def test_high_precision_normal_beats_low_bit_quantization():
    normal = (0.173, -0.419, 0.891)
    fixed = vat.normal_precision(normal)
    legacy = vat.encode_normal(normal, bits=5)
    assert fixed.lighting_delta < legacy.lighting_error
    assert fixed.max_component_error < 1e-4


def test_instance_offsets_are_held_per_instance():
    shared = vat.VATTexture(name="shared")
    first = shared.create_instance("a", frame_offset=11.0)
    second = shared.create_instance("b", frame_offset=22.0)
    checksum = shared.checksum

    first.frame_offset = 999.0
    first.time_offset = 12.5
    first.phase_offset = 0.5

    assert shared.checksum == checksum
    assert shared.verify_checksum()
    assert second.frame_offset == 22.0
    assert first.uniforms["frame_offset"] == 999.0
    assert second.uniforms["frame_offset"] == 22.0

    report = vat.instance_share(shared)
    assert report.isolated is True
    assert report.shared_data_corrupted is False
    assert report.shared_holds_instance_offset is False


def test_lod_switch_inherits_normalized_phase():
    result = vat.lod_switch(0, 1, phase=0.43)
    assert result.inherited is True
    assert result.phase == 0.43
    assert result.new_phase == 0.43
    assert result.phase_delta == 0.0
    assert result.frame_count == 4

    different_count = vat.lod_switch(0, 3, phase=0.456)
    assert different_count.frame_count == 2
    assert math.isclose(different_count.position, 0.912, abs_tol=1e-12)


def test_instance_lod_switch_preserves_playback_phase():
    texture = vat.VATTexture(frame_count=8)
    instance = texture.create_instance(frame_offset=2.0)
    before = instance.playback_phase(time=0.37)
    result = instance.switch_lod(2, frame_count=4, time=0.37)
    after = instance.playback_phase(time=0.37)
    assert result.phase_delta == 0.0
    assert math.isclose(after, before, abs_tol=1e-12)
    assert instance.lod == 2


def test_destroyed_instance_cancels_and_rejects_batch():
    manager = vat.UploadManager()
    texture = vat.VATTexture(name="lifetime")
    manager.register_texture("lifetime", texture)
    instance = manager.create_instance("a", "lifetime")
    batch = manager.create_batch(instance, [vat.UploadItem(frame=0, data=None)])

    destroyed = manager.destroy_instance(instance)
    assert batch.id in destroyed.cancelled
    completion = manager.complete_batch(batch)
    assert completion.ok is False
    assert completion.rejected is True
    assert completion.state == "stale"


def test_stale_batch_cannot_write_after_destroy():
    manager = vat.UploadManager()
    texture = vat.VATTexture(name="write-protect")
    manager.register_texture("write-protect", texture)
    instance = manager.create_instance("a", "write-protect")
    batch = manager.create_batch(
        instance, [{"frame": 0, "data": [[123.0, 123.0, 123.0]]}]
    )
    original = [row[:] for row in texture.positions[0]]

    manager.destroy_instance(instance)
    result = manager.complete_batch(batch)

    assert result.written is False
    assert texture.positions[0] == original
    assert manager.audit().ok


def test_other_instance_batch_survives_destroy():
    report = vat.texture_lifetime()
    assert report.ok is True
    assert report.stale_completion.rejected is True
    assert report.audit.ok is True


def test_instance_generation_is_monotonic_after_same_id_recreate():
    manager = vat.UploadManager()
    old = manager.create_instance("a")
    old_batch = manager.create_batch(old)
    manager.destroy_instance("a")
    recreated = manager.create_instance("a")

    assert recreated is not old
    stale = manager.complete_batch(old_batch)
    assert stale.ok is False and stale.state == "stale"
    fresh = manager.create_batch(recreated)
    assert manager.complete_batch(fresh).ok is True


def test_bounds_wraps_and_reports_frame_rectangle():
    wrapped = vat.bounds(9, frame_count=4, wrap=True)
    assert wrapped.ok and wrapped.frame == 1
    assert wrapped.in_bounds
    assert len(wrapped.center_uv) == 2

    out_of_bounds = vat.bounds(9, frame_count=4, wrap=False)
    assert out_of_bounds.ok is False
    assert out_of_bounds.in_bounds is False


def test_validation_and_audit_reports_are_successful():
    assert vat.validation(1).ok is True
    bad = vat.validation({"frame_count": 0, "normal_bits": 8})
    assert bad.ok is False and bad.errors
    assert vat.audit().ok is True
    assert vat.metrics().ok is True


def test_fallback_keeps_high_precision_normals():
    fallback = vat.fallback("normal_16bit_unsupported")
    assert fallback.ok is True
    assert fallback.use_float_normals is True
    assert fallback.normal_storage.bits_per_channel == 32
    assert fallback.separate_normal_texture is True


def test_blend_scalars_and_vectors():
    assert vat.blend(1.0, 3.0, 0.25) == 1.5
    assert vat.blend([0.0, 2.0], [2.0, 4.0], 0.5) == [1.0, 3.0]
    assert vat.blend((0.0, 2.0), (2.0, 4.0), 0.5) == (1.0, 3.0)
    assert vat.blend(7) == 7
