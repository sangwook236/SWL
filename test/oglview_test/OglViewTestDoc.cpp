// OglViewTestDoc.cpp : implementation of the COglViewTestDoc class
//

#include "stdafx.h"
#include "OglViewTest.h"

#include "OglViewTestDoc.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// COglViewTestDoc

IMPLEMENT_DYNCREATE(COglViewTestDoc, CDocument)

BEGIN_MESSAGE_MAP(COglViewTestDoc, CDocument)
END_MESSAGE_MAP()


// COglViewTestDoc construction/destruction

COglViewTestDoc::COglViewTestDoc()
{
	// TODO: add one-time construction code here

}

COglViewTestDoc::~COglViewTestDoc()
{
}

BOOL COglViewTestDoc::OnNewDocument()
{
	if (!CDocument::OnNewDocument())
		return FALSE;

	// TODO: add reinitialization code here
	// (SDI documents will reuse this document)

	return TRUE;
}




// COglViewTestDoc serialization

void COglViewTestDoc::Serialize(CArchive& ar)
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


// COglViewTestDoc diagnostics

#ifdef _DEBUG
void COglViewTestDoc::AssertValid() const
{
	CDocument::AssertValid();
}

void COglViewTestDoc::Dump(CDumpContext& dc) const
{
	CDocument::Dump(dc);
}
#endif //_DEBUG


// COglViewTestDoc commands
