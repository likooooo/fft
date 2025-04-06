set(PACKAGE_VERSION "1.0.0") # Replace with your actual version

# Check whether the requested version can be exactly matched
if("${PACKAGE_VERSION}" VERSION_EQUAL "${PACKAGE_FIND_VERSION}")
   set(PACKAGE_VERSION_EXACT TRUE)
endif()

# Check whether the found version is compatible
if("${PACKAGE_VERSION}" VERSION_LESS "${PACKAGE_FIND_VERSION}")
   set(PACKAGE_VERSION_COMPATIBLE FALSE)
else()
   set(PACKAGE_VERSION_COMPATIBLE TRUE)
   if ("${PACKAGE_VERSION}" VERSION_EQUAL "${PACKAGE_FIND_VERSION}")
      set(PACKAGE_VERSION_EXACT TRUE)
   endif()
endif()