# import pytest
# from symbiont.vector_dbs.vector_service import ChatContextService


# def test_add_resource_no_resource_type():
#     service = ChatContextService(resource_type=None)
#     with pytest.raises(ValueError, match="Resource document not provided"):
#         service.add_resource()


# def test_add_resource_unsupported_type():
#     service = ChatContextService(resource_type="unsupported")
#     with pytest.raises(ValueError, match="Resource type not supported"):
#         service.add_resource()


# def test_add_resource_pdf(mocker):
#     service = ChatContextService(resource_type="pdf")
#     mocker.patch.object(service, "add_pdf_resource")
#     mocker.patch.object(service, "resource_doc", page_content="some content")
#     service.add_resource()
#     service.add_pdf_resource.assert_called_once()


# def test_add_resource_webpage(mocker):
#     service = ChatContextService(resource_type="webpage")
#     mocker.patch.object(service, "add_web_resource")
#     mocker.patch.object(service, "resource_doc", page_content="some content")
#     service.add_resource()
#     service.add_web_resource.assert_called_once()


# def test_add_resource_youtube(mocker):
#     service = ChatContextService(resource_type="youtube")
#     mocker.patch.object(service, "add_yt_resource")
#     mocker.patch.object(service, "resource_doc", page_content="some content")
#     service.add_resource()
#     service.add_yt_resource.assert_called_once()
