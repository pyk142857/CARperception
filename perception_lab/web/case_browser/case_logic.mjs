export const kinds = {false_negative:'漏检 FN',false_positive:'误检 FP',id_switch:'连续帧 ID 切换',gap_id_change:'间隔后换 ID',center_error_over_1m:'中心误差 >1m'};
export function filterCases(rows, filters) {
 return rows.filter(r => kinds[r.kind] && (!filters.module || r.module===filters.module) && (!filters.kind || r.kind===filters.kind) && (!filters.className || r.class_name===filters.className) && (!filters.query || `${r.case_id} ${r.frame} ${r.tracking_id||''} ${r.instance_token||''}`.includes(filters.query.trim())));
}
export function seekCase(viewer, row) {
 const id=viewer.get_active_recording_id();
 if (!id) throw new Error('记录尚未准备好');
 const range=viewer.get_time_range(id,'frame');
 if (!range || row.frame<range.min || row.frame>range.max) throw new Error('目标帧尚未加载');
 viewer.set_playing(id,false);viewer.set_active_timeline(id,'frame');viewer.set_current_time(id,'frame',row.frame);
 return id;
}
