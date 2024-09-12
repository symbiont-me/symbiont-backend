from symbiont.vector_dbs.chat_context_service import ChatContextService


def test_chat_context_service_initialization():
    resource_doc = "sample_doc"
    resource_identifier = "sample_identifier"
    resource_type = "pdf"
    user_id = "user_123"
    user_query = "sample query"
    study_id = "study_123"

    service = ChatContextService(
        resource_doc=resource_doc,
        resource_identifier=resource_identifier,
        resource_type=resource_type,
        user_id=user_id,
        user_query=user_query,
        study_id=study_id,
    )

    assert service.resource_doc == resource_doc
    assert service.resource_identifier == resource_identifier
    assert service.resource_type == resource_type
    assert service.user_id == user_id
    assert service.user_query == user_query
    assert service.study_id == study_id
