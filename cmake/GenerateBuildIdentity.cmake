execute_process(
  COMMAND git -C "${NINFER_SOURCE_DIR}" rev-parse HEAD
  RESULT_VARIABLE revision_status
  OUTPUT_VARIABLE revision
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
if(NOT revision_status EQUAL 0 OR revision STREQUAL "")
  set(revision "unknown")
endif()

execute_process(
  COMMAND git -C "${NINFER_SOURCE_DIR}" status --porcelain --untracked-files=normal
  RESULT_VARIABLE dirty_status
  OUTPUT_VARIABLE dirty_output
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
if(dirty_status EQUAL 0 AND dirty_output STREQUAL "")
  set(dirty "false")
else()
  set(dirty "true")
endif()

set(contents "#pragma once\n\nnamespace ninfer::build {\ninline constexpr const char* revision = \"${revision}\";\ninline constexpr bool source_dirty = ${dirty};\n}\n")
get_filename_component(output_directory "${NINFER_OUTPUT}" DIRECTORY)
file(MAKE_DIRECTORY "${output_directory}")
set(temporary "${NINFER_OUTPUT}.tmp")
file(WRITE "${temporary}" "${contents}")
execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${temporary}" "${NINFER_OUTPUT}")
file(REMOVE "${temporary}")
