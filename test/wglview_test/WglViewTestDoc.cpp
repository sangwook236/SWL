// WglViewTestDoc.cpp : implementation of the CWglViewTestDoc class
//

#include "stdafx.h"
#include "WglViewTest.h"

#include "WglViewTestDoc.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CWglViewTestDoc

IMPLEMENT_DYNCREATE(CWglViewTestDoc, CDocument)

BEGIN_MESSAGE_MAP(CWglViewTestDoc, CDocument)
END_MESSAGE_MAP()


// CWglViewTestDoc construction/destruction

CWglViewTestDoc::CWglViewTestDoc()
{
	// TODO: add one-time construction code here

}

CWglViewTestDoc::~CWglViewTestDoc()
{
}

BOOL CWglViewTestDoc::OnNewDocument()
{
	if (!CDocument::OnNewDocument())
		return FALSE;

	// TODO: add reinitialization code here
	// (SDI documents will reuse this document)

	return TRUE;
}




// CWglViewTestDoc serialization

void CWglViewTestDoc::Serialize(CArchive& ar)
{
	if (ar.IsStoring())
	{
		// TODO: add storing code here
	}
	else
	{
		// TODO: add loading code here
	}
}


// CWglViewTestDoc diagnostics

#ifdef _DEBUG
void CWglViewTestDoc::AssertValid() const
{
	CDocument::AssertValid();
}

void CWglViewTestDoc::Dump(CDumpContext& dc) const
{
	CDocument::Dump(dc);
}
#endif //_DEBUG


// CWglViewTestDoc commands
