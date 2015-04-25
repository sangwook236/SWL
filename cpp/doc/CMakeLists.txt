#add_subdirectory(reference)
#add_subdirectory(manual)

# add a target to generate API documentation with Doxygen
if(DOXYGEN_FOUND)
	add_custom_target(
		doc_reference
		${DOXYGEN_EXECUTABLE} swl.dox
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		COMMENT "Generating API documentation with Doxygen" VERBATIM
	)
else(DOXYGEN_FOUND)
	add_custom_target(
		doc_reference
		COMMENT "Doxygen Not Found" VERBATIM
	)
endif(DOXYGEN_FOUND)

add_custom_target(
	doc
#	DEPENDS doc_manual doc_reference
	DEPENDS doc_reference
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	COMMENT "Generating documents" VERBATIM
)
