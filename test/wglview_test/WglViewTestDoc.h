// WglViewTestDoc.h : interface of the CWglViewTestDoc class
//


#pragma once


class CWglViewTestDoc : public CDocument
{
protected: // create from serialization only
	CWglViewTestDoc();
	DECLARE_DYNCREATE(CWglViewTestDoc)

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
	virtual ~CWglViewTestDoc();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// Generated message map functions
protected:
	DECLARE_MESSAGE_MAP()
};


