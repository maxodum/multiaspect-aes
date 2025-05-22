#!/usr/bin/env sh
echo "🚀 Initialize Alembic tables..."
alembic revision --autogenerate -m "Initial tables"
echo "🚀 Migrate Alembic tables..."
alembic -c alembic.ini upgrade head
exec "$@"