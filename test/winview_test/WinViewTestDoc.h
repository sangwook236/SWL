// WinViewTestDoc.h : interface of the CWinViewTestDoc class
//


#pragma once


class CWinViewTestDoc : public CDocument
{
protected: // create from serialization only
	CWinViewTestDoc();
	DECLARE_DYNCREATE(CWinViewTestDoc)

// Attributes
public:

// Operations
public:

// Overrides
public:
	virtual BOOL OnNewDocument();
	virtual void Serialize(CArchive& ar);

// Implementation
public:
	virtual ~CWinViewTestDoc();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// Generated message map functions
protected:
	DECLARE_MESSAGE_MAP()
};


