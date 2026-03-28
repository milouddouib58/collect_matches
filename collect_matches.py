# الإضافة الرئيسية: دالة التحقق من صحة البيانات

def validate_row(row: Dict) -> bool:
    """
    ← جديد: التحقق من أن صف البيانات صالح قبل حفظه.
    """
    # يجب أن يكون لديه match_id
    if row.get("match_id") is None:
        return False

    # يجب أن يكون لديه أسماء فرق
    if not row.get("home_team") or not row.get("away_team"):
        return False

    # يجب أن تكون الأهداف أرقام غير سالبة
    hg = row.get("home_goals")
    ag = row.get("away_goals")
    if hg is None or ag is None:
        return False
    try:
        if int(hg) < 0 or int(ag) < 0:
            return False
    except (ValueError, TypeError):
        return False

    # يجب أن يكون لديه تاريخ
    if not row.get("date"):
        return False

    return True


# في دالة collect_league، قبل إضافة الصف:
# ❌ القديم:
#   all_new_rows.append(row)
# ✅ الجديد:
#   if validate_row(row):
#       all_new_rows.append(row)
#   else:
#       print(f"  ⚠️ صف غير صالح تم تجاهله: {row.get('match_id')}")
