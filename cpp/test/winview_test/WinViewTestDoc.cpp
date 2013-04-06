// WinViewTestDoc.cpp : implementation of the CWinViewTestDoc class
//

#include "stdafx.h"
#include "swl/Config.h"
#include "WinViewTest.h"

#include "WinViewTestDoc.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CWinViewTestDoc

IMPLEMENT_DYNCREATE(CWinViewTestDoc, CDocument)

BEGIN_MESSAGE_MAP(CWinViewTestDoc, CDocument)
END_MESSAGE_MAP()


// CWinViewTestDoc construction/destruction

CWinViewTestDoc::CWinViewTestDoc()
{
	// TODO: add one-time construction code here

}

CWinViewTestDoc::~CWinViewTestDoc()
{
}

BOOL CWinViewTestDoc::OnNewDocument()
{
	if (!CDocument::OnNewDocument())
		return FALSE;

	// TODO: add reinitialization code here
	// (SDI documents will reuse this document)

	return TRUE;
}




// CWinViewTestDoc serialization

void CWinViewTestDoc::Serialize(CArchive& ar)
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


// CWinViewTestDoc diagnostics

#ifdef _DEBUG
void CWinViewTestDoc::AssertValid() const
{
	CDocument::AssertValid();
}

void CWinViewTestDoc::Dump(CDumpContext& dc) const
{
	CDocument::Dump(dc);
}
#endif //_DEBUG


// CWinViewTestDoc commands
