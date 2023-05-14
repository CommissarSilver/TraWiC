from utils import santa

coder = santa.SantaCoder()

prefix = """    @distributed_trace
    def list(
        self,
        resource_group_name: str,
        resource_name: str,
        **kwargs: Any
    ) -> "_models.PrivateLinkResourcesListResult":
        \"\"\"Gets a list of private link resources in the specified managed cluster.

        To learn more about private clusters, see:
        https://docs.microsoft.com/azure/aks/private-clusters.

        :param resource_group_name: The name of the resource group. The name is case insensitive.
        :type resource_group_name: str
        :param resource_name: The name of the managed cluster resource.
        :type resource_name: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: PrivateLinkResourcesListResult, or the result of cls(response)
        :rtype: ~azure.mgmt.containerservice.v2022_04_01.models.PrivateLinkResourcesListResult
        :raises: ~azure.core.exceptions.HttpResponseError
        \"\"\"
        cls = kwargs.pop('cls', None)  # type: ClsType["_models.PrivateLinkResourcesListResult"]
"""

suffix = """
        api_version = kwargs.pop('api_version', "2022-04-01")  # type: str

        
        request = build_list_request(
            subscription_id=self._config.subscription_id,
            resource_group_name=resource_group_name,
            resource_name=resource_name,
            api_version=api_version,
            template_url=self.list.metadata['url'],
        )
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)

        pipeline_response = self._client._pipeline.run(  # pylint: disable=protected-access
            request,
            stream=False,
            **kwargs
        )
"""

middle = coder.infill((prefix, suffix), temperature=0.2)

print("\033[92m" + prefix + "\033[93m" + middle + "\033[92m" + suffix)

    # @distributed_trace
    # def list(
    #     self,
    #     resource_group_name: str,
    #     resource_name: str,
    #     **kwargs: Any
    # ) -> "_models.PrivateLinkResourcesListResult":
    #     """Gets a list of private link resources in the specified managed cluster.

    #     To learn more about private clusters, see:
    #     https://docs.microsoft.com/azure/aks/private-clusters.

    #     :param resource_group_name: The name of the resource group. The name is case insensitive.
    #     :type resource_group_name: str
    #     :param resource_name: The name of the managed cluster resource.
    #     :type resource_name: str
    #     :keyword callable cls: A custom type or function that will be passed the direct response
    #     :return: PrivateLinkResourcesListResult, or the result of cls(response)
    #     :rtype: ~azure.mgmt.containerservice.v2022_04_01.models.PrivateLinkResourcesListResult
    #     :raises: ~azure.core.exceptions.HttpResponseError
    #     """
    #     cls = kwargs.pop('cls', None)  # type: ClsType["_models.PrivateLinkResourcesListResult"]
    #     error_map = {
    #         401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError
    #     }
    #     error_map.update(kwargs.pop('error_map', {}))

    #     api_version = kwargs.pop('api_version', "2022-04-01")  # type: str

        
    #     request = build_list_request(
    #         subscription_id=self._config.subscription_id,
    #         resource_group_name=resource_group_name,
    #         resource_name=resource_name,
    #         api_version=api_version,
    #         template_url=self.list.metadata['url'],
    #     )
    #     request = _convert_request(request)
    #     request.url = self._client.format_url(request.url)

    #     pipeline_response = self._client._pipeline.run(  # pylint: disable=protected-access
    #         request,
    #         stream=False,
    #         **kwargs
    #     )
    #     response = pipeline_response.http_response

    #     if response.status_code not in [200]:
    #         map_error(status_code=response.status_code, response=response, error_map=error_map)
    #         raise HttpResponseError(response=response, error_format=ARMErrorFormat)

    #     deserialized = self._deserialize('PrivateLinkResourcesListResult', pipeline_response)

    #     if cls:
    #         return cls(pipeline_response, deserialized, {})

    #     return deserialized

