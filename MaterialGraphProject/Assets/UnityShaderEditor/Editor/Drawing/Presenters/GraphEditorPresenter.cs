﻿using System;
using UnityEditor.MaterialGraph.Drawing.Inspector;
using UnityEngine;
using UnityEngine.Graphing;
using UnityEngine.MaterialGraph;

namespace UnityEditor.MaterialGraph.Drawing
{
    public class GraphEditorPresenter : ScriptableObject
    {
        [SerializeField]
        TitleBarPresenter m_TitleBarPresenter;

        [SerializeField]
        MaterialGraphPresenter m_GraphPresenter;

        [SerializeField]
        GraphInspectorPresenter m_GraphInspectorPresenter;

        public TitleBarPresenter titleBarPresenter
        {
            get { return m_TitleBarPresenter; }
            set { m_TitleBarPresenter = value; }
        }

        public MaterialGraphPresenter graphPresenter
        {
            get { return m_GraphPresenter; }
            set { m_GraphPresenter = value; }
        }

        public GraphInspectorPresenter graphInspectorPresenter
        {
            get { return m_GraphInspectorPresenter; }
            set { m_GraphInspectorPresenter = value; }
        }

        public void Initialize(AbstractMaterialGraph graph, HelperMaterialGraphEditWindow container, string graphName)
        {
            m_TitleBarPresenter = CreateInstance<TitleBarPresenter>();
            m_TitleBarPresenter.Initialize(container);

            m_GraphInspectorPresenter = CreateInstance<GraphInspectorPresenter>();
            m_GraphInspectorPresenter.Initialize(container, graphName);

            m_GraphPresenter = CreateInstance<MaterialGraphPresenter>();
            m_GraphPresenter.Initialize(graph, container);
            m_GraphPresenter.onSelectionChanged += m_GraphInspectorPresenter.UpdateSelection;
        }
    }
}
