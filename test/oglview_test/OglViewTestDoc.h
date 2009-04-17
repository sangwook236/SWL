// OglViewTestDoc.h : interface of the COglViewTestDoc class
//


#pragma once


class COglViewTestDoc : public CDocument
{
protected: // create from serialization only
	COglViewTestDoc();
	DECLARE_DYNCREATE(COglViewTestDoc)

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
	virtual ~COglViewTestDoc();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// Generated message map functions
protected:
	DECLARE_MESSAGE_MAP()
};


