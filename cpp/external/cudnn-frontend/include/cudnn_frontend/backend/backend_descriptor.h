#pragma once

#include <memory>

#include "../graph_helpers.h"
#include "cudnn.h"

namespace cudnn_frontend::detail {

/**
 * @brief RAII wrapper around a `cudnnBackendDescriptor_t` object.
 *
 * This class provides a convenient way to manage the lifetime of a `cudnnBackendDescriptor_t`
 * object using the RAII (Resource Acquisition Is Initialization) idiom. It automatically
 * creates the descriptor when the object is constructed and destroys it when the object
 * is destroyed, ensuring proper resource management and preventing memory leaks.
 *
 * @note The constructor of this class does not throw exceptions. Instead, it stores the
 * status of the descriptor creation operation in the `status` member variable. Callers
 * should check this status and handle any errors accordingly.
 */
class backend_descriptor {
   public:
    /**
     * @brief Constructs a `backend_descriptor` object.
     *
     * @param type The type of the backend descriptor to create.
     */
    backend_descriptor(cudnnBackendDescriptorType_t type) { status = detail::create_descriptor(type, &desc); }

    /**
     * @brief Move constructor.
     *
     * Transfers the ownership of the `cudnnBackendDescriptor_t` object to the new
     * `backend_descriptor` instance.
     *
     * @param other The source `backend_descriptor` object.
     */
    backend_descriptor(backend_descriptor&& other) noexcept : desc(other.desc), status(other.status) {
        other.desc   = nullptr;
        other.status = CUDNN_STATUS_NOT_INITIALIZED;
    }

    /**
     * @brief Move assignment operator.
     *
     * Transfers the ownership of the `cudnnBackendDescriptor_t` object to the new
     * `backend_descriptor` instance.
     *
     * @param other The source `backend_descriptor` object.
     * @return A reference to the current `backend_descriptor` object.
     */
    backend_descriptor&
    operator=(backend_descriptor&& other) noexcept {
        if (this != &other) {
            desc   = other.desc;
            status = other.status;

            other.desc = nullptr;
        }
        return *this;
    }

    /**
     * @brief Destructor.
     *
     * Destroys the `cudnnBackendDescriptor_t` object and frees the associated resources.
     */
    ~backend_descriptor() {
        if (desc) {
            detail::destroy_descriptor(desc);
        }
    }

    /**
     * @brief Deleted copy constructor and assignment operator.
     *
     * `backend_descriptor` objects are not copyable to prevent unintended resource
     * sharing and potential memory leaks.
     */
    backend_descriptor(backend_descriptor const&) = delete;
    backend_descriptor&
    operator=(backend_descriptor const&) = delete;

    /**
     * @brief Initializes a `backend_descriptor` object.
     *
     * @param type The type of the backend descriptor to create.
     */
    error_t
    initialize(cudnnBackendDescriptorType_t type) {
        _CUDNN_CHECK_CUDNN_ERROR(detail::create_descriptor(type, &desc));
        return {error_code_t::OK, ""};
    }

    /**
     * @brief Finalizes a `backend_descriptor` object.
     *
     */
    error_t
    finalize() {
        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(desc));
        return {error_code_t::OK, ""};
    }

    /**
     * @brief Accessor for the underlying `cudnnBackendDescriptor_t` object.
     *
     * @return A const reference to `cudnnBackendDescriptor_t`, the raw pointer to the backend descriptor.
     */
    cudnnBackendDescriptor_t const&
    get_ptr() const {
        return desc;
    }

    /**
     * @brief Accessor for the status of the backend descriptor creation.
     *
     * @return `cudnnStatus_t` The status of the backend descriptor creation operation.
     */
    cudnnStatus_t
    get_status() const {
        return status;
    }

    /**
     * @brief Constructs a default `backend_descriptor` object, but without initializing descriptor
     *
     * Used to return an error code to user for incorrect cuDNN version
     */
    backend_descriptor() = default;

   private:
    cudnnBackendDescriptor_t desc = nullptr;                       //!< Raw pointer to the backend descriptor.
    cudnnStatus_t status          = CUDNN_STATUS_NOT_INITIALIZED;  //!< Status of the descriptor creation operation.
};

}  // namespace cudnn_frontend::detail