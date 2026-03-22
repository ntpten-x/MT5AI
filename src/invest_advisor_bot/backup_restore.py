from __future__ import annotations

import sys
from pathlib import Path

from invest_advisor_bot.bot.backup_manager import BackupManager
from invest_advisor_bot.config import get_settings


def main() -> int:
    settings = get_settings()
    if not settings.database_url.strip():
        print("DATABASE_URL is required to restore backups", file=sys.stderr)
        return 2

    manager = BackupManager(
        backup_dir=settings.backup_dir,
        database_url=settings.database_url,
        retention_days=settings.backup_retention_days,
    )
    backup_path = Path(sys.argv[1]) if len(sys.argv) > 1 else manager.latest_backup_path()
    if backup_path is None:
        print("No backup file found", file=sys.stderr)
        return 1

    manifest = manager.restore_backup(backup_path)
    print(f"Restored backup: {manifest.path}")
    for table_name, count in sorted(manifest.row_counts.items()):
        print(f"- {table_name}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
