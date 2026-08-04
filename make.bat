@echo off
REM Windows shim for the Makefile targets.
REM
REM cmd.exe has no make, and installing one for a handful of short commands is
REM not worth it. This dispatches the same target names to the same commands, so
REM `make test` works here and on a machine that does have make.
REM
REM Keep in step with the Makefile when targets change.

setlocal

if "%~1"=="" goto help
if /I "%~1"=="help"      goto help
if /I "%~1"=="install"   goto install
if /I "%~1"=="lock"      goto lock
if /I "%~1"=="lint"      goto lint
if /I "%~1"=="format"    goto format
if /I "%~1"=="typecheck" goto typecheck
if /I "%~1"=="test"      goto test
if /I "%~1"=="test-fast" goto testfast
if /I "%~1"=="test-data" goto testdata
if /I "%~1"=="render"    goto render
if /I "%~1"=="render-fresh" goto renderfresh
if /I "%~1"=="preview"   goto preview
if /I "%~1"=="check"     goto check
if /I "%~1"=="clean"     goto clean

echo Unknown target: %~1
echo.
goto help

:help
echo   install     Create the environment and install pre-commit hooks
echo   lock        Re-resolve dependencies and update uv.lock
echo   lint        Lint without modifying files
echo   format      Apply lint fixes and formatting
echo   typecheck   Run mypy in strict mode
echo   test        Run the full suite, enforcing the coverage floor
echo   test-fast   Run the suite, skipping tests marked slow
echo   test-data   Run the checks that need the full dataset
echo   render        Render the book, reusing cached results
echo   render-fresh  Re-execute every chapter, ignoring the cache
echo   preview     Render and watch, for writing
echo   check       Everything CI runs
echo   clean       Remove caches and build output
goto end

:install
uv sync --locked || goto fail
uv run pre-commit install || goto fail
goto end

:lock
uv lock || goto fail
goto end

:lint
uv run ruff check . || goto fail
uv run ruff format --check . || goto fail
goto end

:format
uv run ruff check --fix . || goto fail
uv run ruff format . || goto fail
goto end

:typecheck
uv run mypy || goto fail
goto end

:test
uv run pytest --cov-fail-under=80 || goto fail
goto end

:testfast
uv run pytest -m "not slow" --no-cov || goto fail
goto end

:testdata
uv run pytest -m requires_data --no-cov || goto fail
goto end

:render
uv run quarto render || goto fail
goto end

:renderfresh
if exist _freeze rmdir /s /q _freeze
uv run quarto render || goto fail
goto end

:preview
uv run quarto preview || goto fail
goto end

:check
call "%~f0" lint || goto fail
call "%~f0" typecheck || goto fail
call "%~f0" test || goto fail
goto end

:clean
if exist .pytest_cache rmdir /s /q .pytest_cache
if exist .mypy_cache   rmdir /s /q .mypy_cache
if exist .ruff_cache   rmdir /s /q .ruff_cache
if exist htmlcov       rmdir /s /q htmlcov
if exist dist          rmdir /s /q dist
if exist build         rmdir /s /q build
if exist _book         rmdir /s /q _book
if exist .quarto       rmdir /s /q .quarto
if exist .coverage     del /q .coverage
if exist coverage.xml  del /q coverage.xml
for /d /r %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
goto end

:fail
endlocal
exit /b 1

:end
endlocal
exit /b 0
